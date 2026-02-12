run(){

wandb_name=$model

# set to unix time if not set
if [ -z "$SLURM_JOB_ID" ]; then
    SLURM_JOB_ID=$(date +%s)
fi

export NCCL_TIMEOUT=3600
python -u train.py\
    --run_id $SLURM_JOB_ID \
    --model $model \
    --tokenizer $tokenizer \
    --dataset datasets/dolci_think/splits/train_100 \
    --epochs 0 \
    --wandb_name $wandb_name \
    --wandb_group baselines \
    --skip_saving \
    --wandb

python eval.py --run_id $SLURM_JOB_ID --model $model --tokenizer $tokenizer --task aime24
python eval.py --run_id $SLURM_JOB_ID --model $model --tokenizer $tokenizer --task aime25
python eval.py --run_id $SLURM_JOB_ID --model $model --tokenizer $tokenizer --task gpqa --n 4

}


model=allenai/Olmo-3-1025-7B
tokenizer=allenai/Olmo-3-7B-Instruct
run
model=Qwen/Qwen3-8B-Base
tokenizer=Qwen/Qwen3-8B-Base
run
model=Qwen/Qwen3-4B-Base
tokenizer=Qwen/Qwen3-0.6B
run


