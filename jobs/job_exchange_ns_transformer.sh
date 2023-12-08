#!/bin/bash
#SBATCH --job-name=exchange_ns_transformer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=02:00:00
#SBATCH --output="exchange_ns_transformer.txt"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=rg4357@nyu.edu
#SBATCH --account=class
#SBATCH --priority=4294967293

singularity exec --nv --overlay /scratch/rg4357/ns_transformer/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c 'source /ext3/nst.sh;
    python3 -u /scratch/rg4357/ns_transformer/reproduction/Nonstationary_Transformers/run.py \
        --is_training 1 \
        --root_path ../data_provider/dataset/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id Exchange_96_96 \
        --model ns_Transformer \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --gpu 0 \
        --des 'Exp_h16_l2' \
        --p_hidden_dims 16 16 \
        --p_hidden_layers 2 \
        --itr 1;

    python3 -u /scratch/rg4357/ns_transformer/reproduction/Nonstationary_Transformers/run.py \
        --is_training 1 \
        --root_path ../data_provider/dataset/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id Exchange_96_192 \
        --model ns_Transformer \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --gpu 0 \
        --des 'Exp_h16_l2' \
        --p_hidden_dims 16 16 \
        --p_hidden_layers 2 \
        --itr 1;

    python3 -u /scratch/rg4357/ns_transformer/reproduction/Nonstationary_Transformers/run.py \
        --is_training 1 \
        --root_path ../data_provider/dataset/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id Exchange_96_336 \
        --model ns_Transformer \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 336 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --gpu 0 \
        --des 'Exp_h64_l1' \
        --p_hidden_dims 64 \
        --p_hidden_layers 1 \
        --itr 1;


    python3 -u /scratch/rg4357/ns_transformer/reproduction/Nonstationary_Transformers/run.py \
        --is_training 1 \
        --root_path ../data_provider/dataset/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id Exchange_96_720 \
        --model ns_Transformer \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 720 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --gpu 0 \
        --des 'Exp_h64_l2' \
        --p_hidden_dims 64 64 \
        --p_hidden_layers 2 \
        --itr 1;'