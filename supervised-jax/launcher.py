import os
from tpu_pod_launcher import TPUPodClient, TPUPodProject, create_cli

SETUP_SCRIPT = """\
cd ~/
# install basics
apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    curl \
    git \
    vim \
    wget \
    tmux \
    redis-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniforge
rm -rf ~/Miniconda3-py39_4.12.0-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -P ~/
bash ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b

# install dependencies
source ~/miniconda3/bin/activate
conda init bash
conda create -n llama3_train python=3.10 -y
conda activate llama3_train
cd ~/llama3_train
python -m pip install -e .
pip install -U "jax[tpu]==0.4.38" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install tyro flax scalax transformers gcsfs optax wandb
pip install -U transformers==4.47.1 flax==0.10.2

# clean up
cd ~/
rm -rf ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
""".strip()

CHECK_DEVICES = r"""
source ~/miniconda3/bin/activate llama3_train
python -c "import jax; print(jax.devices())"
""".strip()

def check_devices(project: TPUPodProject, verbose: bool=False):
    project.ssh(CHECK_DEVICES, verbose=verbose)

def setup(project: TPUPodProject, verbose: bool=False):
    project.copy(verbose=verbose)
    project.ssh(SETUP_SCRIPT, verbose=verbose)
    project.ssh('mkdir ~/.config/', verbose=verbose)
    project.ssh('mkdir ~/.config/gcloud/', verbose=verbose)
    project.scp('/home/jiayipan/code/25SP/TPU-Train/civic-boulder-204700-3052e43e8c80.json', '~/.config/gcloud/', verbose=verbose)

def debug(project: TPUPodProject, verbose: bool=False):
    import IPython; IPython.embed()

def create_project(tpu_name: str, zone: str) -> TPUPodProject:
    return TPUPodProject(
        client=TPUPodClient(
            tpu_project='civic-boulder-204700',
            tpu_zone=zone,
            user='jiayipan',
            key_path='/home/jiayipan/.ssh/id_rsa',
        ),
        tpu_name=tpu_name,
        copy_dirs=[('/home/jiayipan/code/25SP/LM-Parallel/JAX-Train/llama3_train/', '~/llama3_train/')],
        working_dir='~/llama3_train/',
        copy_excludes=['.git', '__pycache__', '*.pkl', '*.json', '*.jsonl', '*.ipynb'],
        kill_commands=['pkill -9 python'],
    )

if __name__ == "__main__":
    launch_config_path = os.path.join(os.path.dirname(__file__), 'launch_config.json')

    available_tpus = [
        ('jiayi-64-eu', 'europe-west4-a'), # v3-64
        ('jiayi-128-eu', 'europe-west4-a'), # v3-128
        ('jiayi-128-eu-2', 'europe-west4-a'), # v3-128
    ]

    tpu_projects = {name: create_project(name, zone) for name, zone in available_tpus}

    create_cli(
        projects=tpu_projects,
        setup=setup,
        custom_commands={'debug': debug, 'check_devices': check_devices},
        launch_config_path=launch_config_path,
    )
