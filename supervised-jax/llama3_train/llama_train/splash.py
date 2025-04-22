# adapted from: https://github.com/stanford-crfm/levanter/blob/main/src/levanter/models/attention.py
from typing import Optional
import jax.numpy as jnp
import warnings
import jax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as PS
from scalax.sharding import MeshShardingHelper
import functools

# CF https://github.com/google/maxtext/blob/db31dd4b0b686bca4cd7cf940917ec372faa183a/MaxText/layers/attentions.py#L179
def _tpu_splash_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    attention_mask: jnp.ndarray,
    dropout: float = 0.0,
    *,
    attention_dtype: Optional[jnp.dtype] = None,
    block_size: Optional[int] = None,
) -> Optional[jnp.ndarray]:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask

    # Splash attention requires BHSD format
    # We need to reshape the input to match this format
    if dropout != 0.0:
        raise NotImplementedError("Splash attention does not support dropout")

    if attention_dtype is not None and attention_dtype != jnp.float32:
        warnings.warn("Splash attention only supports float32. Switching to float32.")

    attention_dtype = jnp.float32

    B, Sq, Hq, D = query.shape
    Bk, Sk, Hk, Dk = key.shape

    # pre-divide q_ by sqrt(d) to match the reference implementation
    query = query / jnp.sqrt(D)

    # number
    if Sk % 128 != 0:
        raise NotImplementedError(f"Splash attention requires KPos to be a multiple of 128, got {Sk}")

    if block_size is not None and block_size % 128 != 0:
        raise NotImplementedError(f"Splash attention requires block_size to be a multiple of 128, got {block_size}")

    # TODO: must Dk == Dv?
    if key.shape != value.shape:
        raise ValueError("k and v must have the same axes")

    # TODO: this isn't really necessary on TPU?
    if B != Bk:
        raise ValueError(f"Batch axes must be the same for q, k, and v: {B} != {Bk}")

    if D != Dk:
        raise ValueError(f"Embedding axes must be the same for q, k, and v: {D} != {Dk}")

    # MaxText uses a block size of 512
    block_size = block_size or 512

    # copied from MaxText
    @functools.partial(
        shard_map,
        mesh=MeshShardingHelper.get_global_mesh(),
        in_specs=(
            PS(("dp", "fsdp"), "mp", None, None),
            PS(("dp", "fsdp"), "mp", None, None),
            PS(("dp", "fsdp"), "mp", None, None),
            PS(("dp", "fsdp"), None),
        ),
        out_specs=PS(("dp", "fsdp"), "mp", None, None),
        check_rep=False,
    )
    def wrap_flash_attention(q, k, v, attention_mask):
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=min(block_size, Sq),
            block_kv_compute=min(block_size, Sk),
            block_kv=min(block_size, Sk),
            block_q_dkv=min(block_size, Sq),
            block_kv_dkv=min(block_size, Sk),
            block_kv_dkv_compute=min(block_size, Sq),
            block_q_dq=min(block_size, Sq),
            block_kv_dq=min(block_size, Sq),
        )

        segment_ids = splash_attention_kernel.SegmentIds(
            q=attention_mask,
            kv=attention_mask,
        )

        kernel_mask = splash_attention_mask.MultiHeadMask(
            [splash_attention_mask.CausalMask((Sq, Sq)) for _ in range(Hq)],
        )

        # copied from MaxText
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=kernel_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes
        )

        q = q.astype(attention_dtype)
        k = k.astype(attention_dtype)
        v = v.astype(attention_dtype)
        return jax.vmap(splash_kernel)(q, k, v, segment_ids=segment_ids)

    query = query.transpose(0, 2, 1, 3)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    attn_output = wrap_flash_attention(query, key, value, attention_mask)
    attn_output = attn_output.transpose(0, 2, 1, 3)
    return attn_output
