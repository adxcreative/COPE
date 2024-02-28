export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=22226 infer.py \
    --embedding_size 128 \
    --workers 20 \
    --batch_size 10 \
    --checkpoints '/data/phd/SPU/shark/experiments_zrx/cross_domain_emb_rbt6-master/partial_spu_pretrain_cross-domain-in-modal-align/checkpoints/checkpoint_70.pth.tar' \
    --item2fullpath '/data/baixuehan03/datasets/shark/txts/test_all_finish/item2fullpath.pkl'  \
    --item2title '/data/baixuehan03/datasets/shark/txts/test_all_finish/item2title.pkl'  \
    --photo2fullpath '/data/baixuehan03/datasets/shark/txts/test_all_finish/photo2fullpath.pkl' \
    --photo2text '/data/zhaoruixiang/datasets/shark/txts/test_all_finish/baichuan_cleaned_photo2text.pkl' \
    --live2fullpath '/data/baixuehan03/datasets/shark/txts/test_all_finish/liveframe2fullpath.pkl' \
    --live2text '/data/zhaoruixiang/datasets/shark/txts/test_all_finish/baichuan_cleaned_live2text.pkl' \
    --clip_length 8 \
    --output_dir '/data/zhaoruixiang/datasets/shark/txts/test_all_finish/embs/embeddings/shark_temp' \
    --mixed_precision_training \
