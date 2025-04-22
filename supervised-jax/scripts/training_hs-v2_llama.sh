# 12/9/24

# test gpt2 training, this will train a randomly initialized model

# charlie-pod
        # \"tokenizer\": \"gpt2\",
        # \"config\": \"gpt2\"
        # gs://

(
source ~/miniconda3/bin/activate llama3_train
export RUN_NAME="llama-200m-hs-v2"
export GCLOUD_TOKEN_PATH="$HOME/.config/gcloud/civic-boulder-204700-3052e43e8c80.json"
export GCLOUD_PROJECT="civic-boulder-204700"
export HF_TOKEN="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"
export WANDB_API_KEY="53a3e8edb945646eb837622d6422755f5a3131b2"
cd ~/llama3_train
# pip install optax wandb
source ~/miniconda3/bin/activate llama3_train

TRAIN_STEPS=8000
python llama_train_script.py \
    --load_model="paths:{
        \"tokenizer\": \"meta-llama/Llama-2-7b-hf\",
        \"default_config_name\": \"200m\"
    }" \
    --train_data_path="gcs://jiayi-eu/data/hs-v2/train.json" \
    --eval_data_path="gcs://jiayi-eu/data/hs-v2/test.json" \
    --output_dir="gcs://jiayi-eu/lm-parallel-exp/exp-sos/" \
    --sharding="-1,1,1" \
    --num_train_steps=$TRAIN_STEPS \
    --max_length=4096 \
    --bsize=256 \
    --log_freq=100 \
    --num_eval_steps=500 \
    --save_model_freq=100000000 \
    --wandb_project="sos" \
    --param_dtype="fp32" \
    --activation_dtype="fp32" \
    --optim_config="adamw:{
        \"init_lr\": 5e-6,
        \"end_lr\": 5e-7,
        \"lr\": 5e-5,
        \"lr_warmup_steps\": 1,
        \"lr_decay_steps\": $TRAIN_STEPS,
        \"b1\": 0.9,
        \"b2\": 0.999,
        \"clip_gradient\": 1.0,
        \"weight_decay\": 0.01,
        \"bf16_momentum\": false,
        \"multiply_by_parameter_scale\": false,
        \"weight_decay_exclusions\": [],
        \"schedule\": \"cos\",
        \"grad_accum_steps\": 1
    }" \
    --logger_config="{
        \"online\": true,
        \"prefix\": \"$RUN_NAME\",
        \"prefix_to_id\": true
    }" \
    --checkpointer_config="{
        \"save_optimizer_state\": false,
        \"save_float_dtype\": \"bf16\"
    }" \
    --model_config_override="{
        \"bos_token_id\": 1,
        \"eos_token_id\": 2,
        \"pad_token_id\": 0,
        \"remat_block\": \"nothing_saveable\"
    }" \
    --eval_bsize=512 \
    --no_shuffle_train_data \
    --hf_repo_id="LM-Parallel/llama-hs-v2-8k-step"
)


