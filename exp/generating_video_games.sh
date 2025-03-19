#!/bin/bash

#################
# Examples codes to generate video games and animations
# (10min / 25) * 500 = around 3h20min per run (assuming 1 A-100 80Gb)
#################

## method 1) 500 random prompts from the 10K prompts of the Huggingface dataset
# High precision
CUDA_VISIBLE_DEVICES=0 python3 generate.py \
 --output_dir=/home/mygames/500_minp0.05_t0.1 \
 --n_answers_per_prompts=1 --max_num_seqs=25 --max_model_len=32768 \
 --prompts_hf=SamsungSAILMontreal/GameAnimationPrompts10K \
 --temperature=0.1 --min_p=0.05 \
 --max_n_prompts=500 \
 --randomize_order=True
# Precise, but more diverse (recommended setting, but we haven't tried many settings, so feel free to experiment)
CUDA_VISIBLE_DEVICES=0 python3 generate.py \
 --output_dir=/home/mygames/500_minp0.05_t0.1 \
 --n_answers_per_prompts=1 --max_num_seqs=25 --max_model_len=32768 \
 --prompts_hf=SamsungSAILMontreal/GameAnimationPrompts10K \
 --temperature=0.1 --min_p=0.05 \
 --max_n_prompts=500 \
 --randomize_order=True

## method 2) inline prompts (seperated by Ŧ)
# High precision
CUDA_VISIBLE_DEVICES=0 python3 generate.py \
 --output_dir=/home/mygames/500_minp0.05_t0.1 \
 --n_answers_per_prompts=1 --max_num_seqs=25 --max_model_len=32768 \
 --prompts='Make me a game about a horseŦMake a game about a turtle' \
 --temperature=0.1 --min_p=0.05
# Precise, but more diverse (recommended setting, but we haven't tried many settings, so feel free to experiment)
CUDA_VISIBLE_DEVICES=0 python3 generate.py \
 --output_dir=/home/mygames/500_minp0.05_t0.1 \
 --n_answers_per_prompts=1 --max_num_seqs=25 --max_model_len=32768 \
 --prompts='Make me a game about a horseŦMake a game about a turtle' \
 --temperature=0.1 --min_p=0.05

## method 3) file containing prompts (seperated by \nŦ\n)
# Example: example_prompt_file.txt
# -------------
# Generate me blablabla.
# Ŧ
# Generate me this other thing that
# looks like a turtle.
# Ŧ
# Generate me this final thing.
# -------------
# High precision
CUDA_VISIBLE_DEVICES=0 python3 generate.py \
 --output_dir=/home/mygames/500_minp0.05_t0.1 \
 --n_answers_per_prompts=1 --max_num_seqs=25 --max_model_len=32768 \
 --prompt_file='example_prompt_file.txt' \
 --temperature=0.1 --min_p=0.05
# Precise, but more diverse (recommended setting, but we haven't tried many settings, so feel free to experiment)
CUDA_VISIBLE_DEVICES=0 python3 generate.py \
 --output_dir=/home/mygames/500_minp0.05_t0.1 \
 --n_answers_per_prompts=1 --max_num_seqs=25 --max_model_len=32768 \
 --prompt_file='example_prompt_file.txt' \
 --temperature=0.1 --min_p=0.05
