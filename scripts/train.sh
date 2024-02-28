# --use_item_text_photo_cross 0.2 \
# --use_item_text_live_cross 0.2 \
# --use_item_visual_photo_cross 0.2 \
# --use_item_visual_live_cross 0.2 \
# --use_item_photo_text_cross 0.2 \
# --use_item_photo_visual_cross 0.2 \
# --use_item_live_text_cross 0.2 \
# --use_item_live_visual_cross 0.2 \
# --use_photo_text_live_cross 0.2 \
# --use_photo_visual_live_cross 0.2 \
# --use_photo_live_text_cross 0.2 \
# --use_photo_live_visual_cross 0.2 \
# --use_item_text_photo_text_cross 1 \
# --use_item_text_live_text_cross 1 \
# --use_photo_text_live_text_cross 1 \
# --use_item_visual_photo_visual_cross 1 \
# --use_item_visual_live_visual_cross 1 \
# --use_photo_visual_live_visual_cross 1 \
# --use_item_photo_cross 1 \
# --use_item_live_cross 1 \
# --use_photo_live_cross 1 \

export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="11.37.25.219" --master_port=22223 train.py \
    --embedding_size 128 \
    --workers 20 \
    --epochs 100 \
    --warm_up_iters 100 \
    --max_iters 5000 \
    --lr_freq 10 \
    --batch_size 8 \
    --lr 5e-3 \
    --text_lr_factor 1e-3 \
    --visual_lr_factor 2e-3 \
    --fusion_lr_factor 1.0 \
    --other_lr_factor 1.0 \
    --mixed_precision_training \
    --clip_length 8 \
    --finetune \
    --use_item_text_item_visual_cross 1.0 \
    --use_photo_text_photo_visual_cross 1.0 \
    --use_live_text_live_visual_cross 1.0 \
    --use_wandb_recorder \
    --train_file '/data/zhaoruixiang/datasets/shark/test/train_all/lmdbs/partial_spus.txt' \
    --spu2item_lmdb '/data/baixuehan03/datasets/shark/test/train_all/lmdbs/spu2item' \
    --spu2photo_lmdb '/data/baixuehan03/datasets/shark/test/train_all/lmdbs/spu2photo' \
    --spu2live_lmdb '/data/baixuehan03/datasets/shark/test/train_all/lmdbs/spu2live' \
    --clip2live_lmdb '/data/baixuehan03/datasets/shark/test/train_all/lmdbs/clip2live' \
    --live2texts_lmdb '/data/zhaoruixiang/datasets/shark/test/train_all/lmdbs/baichuan_cleaned_live2text_partial' \
    --photo2frames_lmdb '/data/baixuehan03/datasets/shark/test/train_all/lmdbs/photo2frames' \
    --photo2texts_lmdb '/data/zhaoruixiang/datasets/shark/test/train_all/lmdbs/baichuan_cleaned_photo2text_partial' \
    --item_imgs_lmdb '/data/baixuehan03/datasets/shark/test/train_all/lmdbs/item_imgs' \
    --item_titles_lmdb '/data/baixuehan03/datasets/shark/test/train_all/lmdbs/item_titles' \
    --photo2fullpath_lmdb '/data/baixuehan03/datasets/shark/test/train_all/lmdbs/photo2fullpath' \
    --live2fullpath_lmdb '/data/baixuehan03/datasets/shark/test/train_all/lmdbs/live2fullpath' \
    --output_dir '/data/phd/SPU/shark/experiments_zrx/cross_domain_emb_rbt6-master/partial_spu_pretrain_cross-modal-in-domain-align'
