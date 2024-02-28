source /opt/conda/bin/activate
conda activate baichuan
cd /data/zhaoruixiang/code/shark_zrx
CUDA_VISIBLE_DEVICES=7 python preprocess/baichuan_train_live_asr_clean.py --index=36