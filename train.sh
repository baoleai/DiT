PJRT_DEVICE='CUDA' CUDA_VISIBLE_DEVICES=7 torchrun --nnodes=1 --nproc_per_node=1 train.py --model DiT-XL/2 --data-path /home/baole.abl/baole/github/imagenet/train --global-batch-size=96
