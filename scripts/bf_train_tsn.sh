python main.py \
    --task long \
    --seg --anticipate --pos_emb\
    --n_encoder_layer 2 --n_decoder_layer 1 --batch_size 3 --hidden_dim 128 --max_pos_len 2000\
    --epochs 60 --mode=train --input_type=TSN --split=$1  --input_dim=1024\
    --dataset breakfast 2>&1 | tee -a bf_tsn_log.txt

