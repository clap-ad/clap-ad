export PYTHONPATH=$PYTHONPATH:/path/to/Bench2Drive-Speed/TCP/TCP
CUDA_VISIBLE_DEVICES=0 python TCP/train_speed_virtual.py --id TCP-B2DS-virtual --gpus 1 --epochs 60 --batch_size 300