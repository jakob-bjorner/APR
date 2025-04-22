# 12/9/24

# test gpt2 training, this will train a randomly initialized model

# charlie-pod
        # \"tokenizer\": \"gpt2\",
        # \"config\": \"gpt2\"
        # gs://

(
source ~/miniconda3/bin/activate llama3_train
pip install -U "jax[tpu]==0.4.38" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
export RUN_NAME="hsp_v0_500k"
export GCLOUD_TOKEN_PATH="$HOME/.config/gcloud/civic-boulder-204700-3052e43e8c80.json"
export GCLOUD_PROJECT="civic-boulder-204700"
export HF_TOKEN="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"
export WANDB_API_KEY="0929e692448f1bc929d71d7e3ece80073c3041e6"
cd ~/llama3_train
# pip install optax wandb
source ~/miniconda3/bin/activate llama3_train

TRAIN_STEPS=19000
python gpt2_train_script.py \
    --load_model="paths:{
        \"tokenizer\": \"gpt2\",
        \"config\": \"LM-Parallel/jax-reference-gp2-s\"
    }" \
    --train_data_path="gcs://jiayi-eu/data/hsp_v0_500k/train.json" \
    --eval_data_path="gcs://jiayi-eu/data/hsp_v0_500k/test.json" \
    --output_dir="gcs://jiayi-eu/lm-parallel-exp/exp-hsp/" \
    --sharding="-1,1,1" \
    --num_train_steps=$TRAIN_STEPS \
    --max_length=4096 \
    --bsize=256 \
    --log_freq=512 \
    --num_eval_steps=512 \
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
        \"clip_gradient\": 100.0,
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
        \"bos_token_id\": 50256,
        \"eos_token_id\": 50256,
        \"pad_token_id\": 50256,
        \"remat_block\": \"nothing_saveable\",
        \"resid_pdrop\": 0.00,
        \"embd_pdrop\": 0.00,
        \"attn_pdrop\": 0.00,
        \"n_positions\": 4096
    }" \
    --eval_bsize=512 \
    --no_shuffle_train_data \
    --hf_repo_id="LM-Parallel/hsp-v0"
)


