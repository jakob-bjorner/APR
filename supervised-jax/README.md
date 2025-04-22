# Example Supervised Training Instructions
> Fork from [JAX_llama](https://github.com/Sea-Snell/JAX_llama)

First do the following:

To orchestrate the TPU pod
```
pip install git+https://github.com/Sea-Snell/tpu_pod_launcher.git@main
```

There is a config in training_run.sh and a launcher defined in `launcher.py`.

The launcher basically uses a tool to run the installation and training script on all hosts in the pod at once.

You will probably want to edit the launcher's `available_tpus` to reflect the TPUs you have access to.

You will also need a google cloud token in a json file somewhere, which has write permissions to buckets. You should change the path in `line 51` in the launcher to point to this file.

You may also want to modify the ssh info and copy path in `lines 56-69` in the launcher.

Make sure you set the API keys correctly at the top of the `training_run.sh` script. And also edit the bucket paths as needed.

The data should be stored in a json list of strings:

```
[
    "seq1",
    "seq2",
    ...
]
```

Use the `to_dataset.ipynb` notebook to convert the dataset to the correct format and upload to GCS.


To install all dependencies on the TPU hosts run:

```
python launcher.py setup --project=jiayi-128-eu
-2
```

You only need to do this once for each TPU pod.

where you_tpu_name refers to the name of the TPU in the `available_tpus` list in the launcher.

To launch the training run:

```
conda activate GPML
cd /home/jiayipan/code/25SP/LM-Parallel/JAX-Train
python launcher.py launch training_sos-split-digit-v2.sh --project=jiayi-128-eu
python launcher.py launch training_hsp_2x.sh --project=jiayi-128-eu
python launcher.py launch training_run.sh --project=jiayi-128-eu
python launcher.py launch scripts/training_sos_llama_10ep_v2.sh --project=jiayi-128-eu
python launcher.py launch scripts/hsp-v3.sh --project=jiayi-128-eu
python launcher.py launch scripts/hs-v3.sh --project=jiayi-128-eu-2
python launcher.py launch scripts/sos-v3.sh --project=jiayi-64-eu

This will: 1) copy the latest version of `llama3_train` to the TPUs; 2) stop anything running on the TPUs; 3) run the training script on the TPUs.

To print the output of the training run, you can run:

```
python launcher.py check --project=your_tpu_name
python launcher.py check --project=jiayi-128-eu
```

To terminate an ongoing training run, you can run:

```
python launcher.py stop --project=your_tpu_name
```

The 3 mesh dimensions in the config currently correspond to (replica,fsdp,tensor). We can also in add a sequence parallel dimension without much difficulty if needed.

### Test Model
Quickly test the model with FLAX/JAX
https://colab.research.google.com/drive/1X7ElvcwrAk5nt_dkAsZUZo6IOp4nsW3L?usp=sharing

### Export To PyTorch
You can use this script to easily export the model to pytorch (huggingface compatible).

https://colab.research.google.com/drive/1XD3bJ1PQuHKwSO2cB0NCc6BkyJoptF19?usp=sharing

