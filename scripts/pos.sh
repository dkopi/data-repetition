run(){

dataset=dakopi/numina_8b_pos__train_${samples}
wandb_name=s${samples}_e${epochs}

# set to unix time if not set
if [ -z "$SLURM_JOB_ID" ]; then
    SLURM_JOB_ID=$(date +%s)
fi

export NCCL_TIMEOUT=3600
python -u train.py\
    --run_id $SLURM_JOB_ID \
    --model $model \
    --tokenizer $tokenizer \
    --dataset $dataset \
    --epochs $epochs \
    --val_set $val_set \
    --lr $lr \
    --wandb_name $wandb_name \
    --wandb_group $wandb_group \
    --wandb

python eval.py --run_id $SLURM_JOB_ID --model ckpts/$SLURM_JOB_ID --tokenizer $tokenizer --task aime24
python eval.py --run_id $SLURM_JOB_ID --model ckpts/$SLURM_JOB_ID --tokenizer $tokenizer --task aime25
python eval.py --run_id $SLURM_JOB_ID --model ckpts/$SLURM_JOB_ID --tokenizer $tokenizer --task gpqa --n 4

}



model=allenai/Olmo-3-1025-7B
tokenizer=allenai/Olmo-3-7B-Instruct

wandb_group=pos_olmo__8b
val_set=dakopi/numina_8b_pos__val

lr=2e-5



for epochs in 32; do
for samples in 200; do
sleep 2
run
done
done

for epochs in 16; do
for samples in 200 400; do
sleep 2
run
done
done

for epochs in 8; do
for samples in 200 400 800; do
sleep 2
run
done
done

for epochs in 4; do
for samples in 200 400 800 1600; do
sleep 2
run
done
done

for epochs in 2; do
for samples in 200 400 800 1600 3200; do
sleep 2
run
done
done

for epochs in 1; do
for samples in 200 400 800 1600 3200 6400; do
sleep 2
run
done
done


