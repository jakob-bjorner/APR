from typing import Optional, Any, List
import os
import gcsfs
import jax
from flax.serialization import from_bytes, to_state_dict, from_state_dict, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict, empty_node
import msgpack
import jax.numpy as jnp
from functools import partial
import re
from optax import softmax_cross_entropy_with_integer_labels
import wandb
import uuid
from socket import gethostname
import tempfile
from jax import lax

GCLOUD_TOKEN_PATH = os.environ.get('GCLOUD_TOKEN_PATH', None)
GCLOUD_PROJECT = os.environ.get('GCLOUD_PROJECT', None)

def open_with_bucket(
    path: Any, 
    mode: str="rb", 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
    **kwargs, 
):
    # backup to env vars if None
    if gcloud_project is None:
        gcloud_project = GCLOUD_PROJECT
    if gcloud_token is None:
        gcloud_token = GCLOUD_TOKEN_PATH
    # load from google cloud storage if starts with "gcs://"
    if path.startswith('gcs://'):
        f = gcsfs.GCSFileSystem(project=gcloud_project, token=gcloud_token).open(path[len('gcs://'):], mode=mode, **kwargs)
    else:
        f = open(path, mode=mode, **kwargs)
    return f

def delete_with_bucket(
    path: str, 
    recursive: bool=True, 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
) -> None:
    # backup to env vars if None
    if gcloud_project is None:
        gcloud_project = GCLOUD_PROJECT
    if gcloud_token is None:
        gcloud_token = GCLOUD_TOKEN_PATH
    # delete from google cloud storage if starts with "gcs://"
    if path.startswith('gcs://'):
        path = path[len('gcs://'):]
        gcsfs.GCSFileSystem(project=gcloud_project, token=gcloud_token).rm(path, recursive=recursive)
    else:
        os.system(f"rm -{'r' if recursive else ''}f {path}")

def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]

def get_float_dtype_by_name(dtype):
    return {
        'bf16': jnp.bfloat16,
        'bfloat16': jnp.bfloat16,
        'fp16': jnp.float16,
        'float16': jnp.float16,
        'fp32': jnp.float32,
        'float32': jnp.float32,
        'fp64': jnp.float64,
        'float64': jnp.float64,
    }[dtype]


def float_tensor_to_dtype(tensor, dtype):
    if dtype is None or dtype == '':
        return tensor
    if isinstance(dtype, str):
        dtype = get_float_dtype_by_name(dtype)
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    if getattr(tensor, 'dtype', None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


def float_to_dtype(tree, dtype):
    return jax.tree_util.tree_map(
        partial(float_tensor_to_dtype, dtype=dtype), tree
    )

def load_checkpoint(path, target=None, shard_fns=None, remove_dict_prefix=None, convert_to_dtypes=None):
    if shard_fns is not None:
        shard_fns = flatten_dict(
            to_state_dict(shard_fns)
        )
    if convert_to_dtypes is not None:
        convert_to_dtypes = flatten_dict(
            to_state_dict(convert_to_dtypes)
        )
    if remove_dict_prefix is not None:
        remove_dict_prefix = tuple(remove_dict_prefix)
    flattend_train_state = {}
    with open_with_bucket(path, 'rb') as fin:
        # 83886080 bytes = 80 MB, which is 16 blocks on GCS
        unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=0)
        for key, value in unpacker:
            key = tuple(key)
            if remove_dict_prefix is not None:
                if key[:len(remove_dict_prefix)] == remove_dict_prefix:
                    key = key[len(remove_dict_prefix):]
                else:
                    continue

            tensor = from_bytes(None, value)
            if convert_to_dtypes is not None:
                tensor = float_tensor_to_dtype(tensor, convert_to_dtypes[key])
            if shard_fns is not None:
                tensor = shard_fns[key](tensor)
            flattend_train_state[key] = tensor

    if target is not None:
        flattened_target = flatten_dict(
            to_state_dict(target), keep_empty_nodes=True
        )
        for key, value in flattened_target.items():
            if key not in flattend_train_state and value == empty_node:
                flattend_train_state[key] = value

    train_state = unflatten_dict(flattend_train_state)
    if target is None:
        return train_state

    return from_state_dict(target, train_state)

def save_checkpoint(train_state, path, gather_fns=None, float_dtype=None):
    train_state = to_state_dict(train_state)
    packer = msgpack.Packer()
    flattend_train_state = flatten_dict(train_state)
    if gather_fns is not None:
        gather_fns = flatten_dict(to_state_dict(gather_fns))

    with open_with_bucket(path, "wb") as fout:
        for key, value in flattend_train_state.items():
            if gather_fns is not None:
                value = gather_fns[key](value)
            value = float_tensor_to_dtype(value, float_dtype)
            fout.write(packer.pack((key, to_bytes(value))))

def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)

def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )

def get_weight_decay_mask(exclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """
    def decay(name, _):
        for rule in exclusions:
            if re.search(rule, name) is not None:
                return False
        return True

    def weight_decay_mask(params):
        return named_tree_map(decay, params, sep='/')

    return weight_decay_mask

def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    logits = logits.astype(jnp.float32) # for numerical stability
    token_loss = softmax_cross_entropy_with_integer_labels(logits, tokens)
    loss = jnp.mean(token_loss, where=valid > 0.0)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == tokens, where=valid > 0.0)
    return loss, accuracy

def global_norm(tree):
    """ Return the global L2 norm of a pytree. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))

def average_metrics(metrics):
    return jax.tree_map(
        lambda *args: jnp.mean(jnp.stack(args)),
        *metrics
    )

def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, dict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output

# logger adapted from mlxu: https://github.com/young-geng/mlxu/blob/main/mlxu/logging.py

class WandbLogger:
    def __init__(
        self,
        project: str,
        output_dir: Optional[str]=None,
        config_to_log: Optional[Any]=None,
        enable: bool=True,
        online: bool=False,
        prefix: Optional[str]=None,
        experiment_id: Optional[str]=None,
        wandb_dir: Optional[str]=None,
        notes: Optional[str]=None,
        entity: Optional[str]=None,
        prefix_to_id: bool=False,
    ):
        self.enable = enable
        self.notes = notes
        self.entity = entity
        self.project = project
        self.online = online
        self.experiment_id = experiment_id

        if self.experiment_id is None:
            self.experiment_id = f'{uuid.uuid4().hex}-{uuid.uuid1().hex}'
        if prefix is not None:
            if prefix_to_id:
                self.experiment_id = f"{prefix}--{self.experiment_id}"
            else:
                self.project = f"{prefix}--{self.project}"
        
        self.wandb_dir = wandb_dir
        self.output_dir = output_dir
        if self.enable:
            if self.output_dir is not None:
                self.output_dir = os.path.join(self.output_dir, self.experiment_id)
                if not self.output_dir.startswith('gcs://'):
                    os.makedirs(self.output_dir, exist_ok=True)
            if self.wandb_dir is None:
                if (self.output_dir is not None) and (not self.output_dir.startswith('gcs://')):
                    self.wandb_dir = self.output_dir
            else:
                assert not self.wandb_dir.startswith('gcs://')
                self.wandb_dir = os.path.join(self.wandb_dir, self.experiment_id)
                os.makedirs(self.wandb_dir, exist_ok=True)
        
        if config_to_log is not None:
            self.config_to_log = flatten_config_dict(config_to_log)
            if "hostname" not in self.config_to_log:
                self.config_to_log["hostname"] = gethostname()
            if "experiment_id" not in self.config_to_log:
                self.config_to_log["experiment_id"] = self.experiment_id
            if "logger_output_dir" not in self.config_to_log:
                self.config_to_log["logger_output_dir"] = self.output_dir
            if "wandb_dir" not in self.config_to_log:
                self.config_to_log["wandb_dir"] = self.wandb_dir
        else:
            self.config_to_log = None
        
        if self.enable:
            self.run = wandb.init(
                reinit=True,
                config=self.config_to_log,
                project=self.project,
                dir=self.wandb_dir,
                id=self.experiment_id,
                notes=self.notes,
                entity=self.entity,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
                mode="online" if self.online else "offline",
            )
        else:
            self.run = None
    
    def log(self, *args, **kwargs):
        if self.enable:
            self.run.log(*args, **kwargs)
    
    def finish(self):
        if self.enable:
            wandb.finish()
    
    def can_save(self) -> bool:
        return self.enable and (self.output_dir is not None)

def jax_distributed_initalize(
    initialize_jax_distributed: bool=False,
    local_device_ids: Optional[List[int]]=None,
    coordinator_address: Optional[str]=None,
    num_processes: Optional[int]=None,
    process_id: Optional[int]=None,
):
    if initialize_jax_distributed:
        if local_device_ids is not None:
            local_device_ids = [int(x) for x in local_device_ids.split(',')]
        else:
            local_device_ids = None

        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
            local_device_ids=local_device_ids,
        )

def jax_distributed_barrier():
    # Dummy function that all processes run
    def computation(x):
        result = x * x
        return result

    @partial(jax.pmap, axis_name='i')
    def sync_barrier(x):
        # Perform a trivial collective operation, acting as a barrier
        c = lax.psum(x, axis_name='i')
        return computation(x) + computation(c)

    # Dummy input
    x = jnp.ones((jax.local_device_count(),))

    # Run the barrier + computation
    results = sync_barrier(x)

    jax.block_until_ready(results)
