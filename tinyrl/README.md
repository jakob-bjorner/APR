# Installation
```
conda create -n grpo python=3.10
# install sgl
pip install --upgrade pip
pip install sgl-kernel --force-reinstall --no-deps
pip install "sglang[all]>=0.4.3.post1" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# other utils
pip install hydra-core omegaconf wandb
```

## For HSP, SGLang needs to be patched
Remove this check in `python/sglang/srt/managers/tokenizer_manager.py` in your local SGLang repo:
```
        # if (
        #     obj.sampling_params.get("max_new_tokens") is not None
        #     and obj.sampling_params.get("max_new_tokens") + input_token_num
        #     >= self.context_len
        # ):
        #     raise ValueError(
        #         f"Requested token count exceeds the model's maximum context length "
        #         f"of {self.context_len} tokens. You requested a total of "
        #         f"{obj.sampling_params.get('max_new_tokens') + input_token_num} "
        #         f"tokens: {input_token_num} tokens from the input messages and "
        #         f"{obj.sampling_params.get('max_new_tokens')} tokens for the "
        #         f"completion. Please reduce the number of tokens in the input "
        #         f"messages or the completion to fit within the limit."
        #     )
```

This file is at https://github.com/sgl-project/sglang/blob/45205d88a08606d5875476fbbbc76815a5107edd/python/sglang/srt/managers/tokenizer_manager.py#L350

# Data Preparation
Please put `hs_train_prefix.json`, `hs_val_prefix.json`, `hsp_train_beam10_subbeam15_prefix.json`, and `hsp_val_prefix.json` in the `data` folder.

You can download them from [https://huggingface.co/datasets/Parallel-Reasoning/apr_rl_data](https://huggingface.co/datasets/Parallel-Reasoning/apr_rl_data).

# Run
Each run requires two GPUs: one for model training and one for serving with SGLang.

## RL on HSP without subcall condition
```
export CUDA_VISIBLE_DEVICES=0,1
python trainer.py --config-name hsp
```

Reference checkpoint: [https://huggingface.co/Parallel-Reasoning/hsp_grpo](https://huggingface.co/Parallel-Reasoning/hsp_grpo)

## RL on HSP with subcall condition (condition set to 10)
```
export CUDA_VISIBLE_DEVICES=0,1
python trainer.py --config-name hsp_cond10
```

Reference checkpoint: [https://huggingface.co/Parallel-Reasoning/hsp_cond10_grpo](https://huggingface.co/Parallel-Reasoning/hsp_cond10_grpo)

## RL on HS
```
export CUDA_VISIBLE_DEVICES=0,1
python trainer.py --config-name hs
```

Reference checkpoint: [https://huggingface.co/Parallel-Reasoning/hs_grpo](https://huggingface.co/Parallel-Reasoning/hs_grpo)

You can set your own config files and specify different configs with `--config-name <config_name>`.
