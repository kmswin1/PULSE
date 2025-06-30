# When you train with 8 GPUs.
torchrun --nproc_per_node=8 \
    train.py $MODEL_NAME $DATA_NAME