python -m torch.distributed.launch --nproc_per_node=3 train0.py
python -m torch.distributed.launch --nproc_per_node=3 train1.py
python -m torch.distributed.launch --nproc_per_node=3 train2.py
python -m torch.distributed.launch --nproc_per_node=3 train3.py
python -m torch.distributed.launch --nproc_per_node=3 train4.py
python -m torch.distributed.launch --nproc_per_node=3 train5.py
python valid5.py
