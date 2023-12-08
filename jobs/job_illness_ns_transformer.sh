#!/bin/bash
#SBATCH --job-name=illness_ns_transformer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=01:30:00
#SBATCH --output="illness_ns_transformer.txt"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=rg4357@nyu.edu
#SBATCH --account=class
#SBATCH --priority=4294967293

singularity exec --nv --overlay /scratch/rg4357/ns_transformer/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c 'source /ext3/nst.sh;
    python3 -u /scratch/rg4357/ns_transformer/reproduction/Nonstationary_Transformers/run.py \
        --is_training 1 \
        --root_path ../data_provider/dataset/illness/ \
        --data_path national_illness.csv \
        --model_id ili_36_24 \
        --model ns_Transformer \
        --data custom \
        --features M \
        --seq_len 36 \
        --label_len 18 \
        --pred_len 24 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --gpu 0 \
        --des 'Exp_h32_l2' \
        --p_hidden_dims 32 32 \
        --p_hidden_layers 2 \
        --itr 1;

    python3 -u /scratch/rg4357/ns_transformer/reproduction/Nonstationary_Transformers/run.py \
        --is_training 1 \
        --root_path ../data_provider/dataset/illness/ \
        --data_path national_illness.csv \
        --model_id ili_36_36 \
        --model ns_Transformer \
        --data custom \
        --features M \
        --seq_len 36 \
        --label_len 18 \
        --pred_len 36 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --gpu 0 \
        --des 'Exp_h32_l2' \
        --p_hidden_dims 32 32 \
        --p_hidden_layers 2 \
        --itr 1;

    python3 -u /scratch/rg4357/ns_transformer/reproduction/Nonstationary_Transformers/run.py \
        --is_training 1 \
        --root_path ../data_provider/dataset/illness/ \
        --data_path national_illness.csv \
        --model_id ili_36_48 \
        --model ns_Transformer \
        --data custom \
        --features M \
        --seq_len 36 \
        --label_len 18 \
        --pred_len 48 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --gpu 0 \
        --des 'Exp_h16_l2' \
        --p_hidden_dims 16 16 \
        --p_hidden_layers 2 \
        --itr 1;

    python3 -u /scratch/rg4357/ns_transformer/reproduction/Nonstationary_Transformers/run.py \
        --is_training 1 \
        --root_path ../data_provider/dataset/illness/ \
        --data_path national_illness.csv \
        --model_id ili_36_60 \
        --model ns_Transformer \
        --data custom \
        --features M \
        --seq_len 36 \
        --label_len 18 \
        --pred_len 60 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --gpu 0 \
        --des 'Exp_h8_l2' \
        --p_hidden_dims 8 8 \
        --p_hidden_layers 2 \
        --itr 1;'
