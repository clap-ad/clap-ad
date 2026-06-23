export PYTHONPATH=$PYTHONPATH:/path/to/Bench2Drive-Speed/TCP/TCP
CUDA_VISIBLE_DEVICES=0 python TCP/train_speed.py --id TCP-B2DS --gpus 1 --epochs 60 --batch_size 300