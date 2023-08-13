python main.py --task long --seg --anticipate --pos_emb --n_query 20\
    --n_encoder_layer 2 --n_decoder_layer 2 --batch_size 1 --hidden_dim 512\
	--max_pos_len 10000 --sample_rate 6 --epochs 70 --mode=train --input_type=TSN \
	--split=$1 --input_dim=1024 --dataset ek55 2>&1 | tee -a ek55_tsn_log.txt
