export PYTHONPATH=$PYTHONPATH:/path/to/Bench2Drive-Speed/TCP/TCP
CUDA_VISIBLE_DEVICES=1 python TCP/train.py --id TCP --gpus 1 --epochs 60 --batch_size 300