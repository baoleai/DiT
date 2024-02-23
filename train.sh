export XLA_FLAGS='--xla_multiheap_size_constraint_per_heap=15032385536'
export XLA_IR_SHAPE_CACHE_SIZE=100000000
export XLA_ALLOCATOR_FRACTION=0.95

PJRT_DEVICE='CUDA' torchrun --nnodes=1 --nproc_per_node=1 train_acc.py --model DiT-XL/2 --data-path /home/baole.abl/baole/github/imagenet/train --global-batch-size=64
