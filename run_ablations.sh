#!/bin/bash

# stable29
source ~/stable29/bin/activate
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt.py; sleep 15; done
mv logs stable29-train_gpt_logs
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_pr146.py; sleep 15; done
mv logs stable29-train_gpt_pr146_logs

# dev0926 (with extra ablation scripts)
source ~/dev0926/bin/activate
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt.py; sleep 15; done
mv logs dev0926-train_gpt_logs
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_pr146.py; sleep 15; done
mv logs dev0926-train_gpt_pr146_logs
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_1dim_ablation.py; sleep 15; done
mv logs dev0926-train_gpt_1dim_ablation_logs
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_attn_reshape_ablation.py; sleep 15; done
mv logs dev0926-train_gpt_attn_reshape_ablation_logs
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_lr_mul_ablation.py; sleep 15; done
mv logs dev0926-train_gpt_lr_mul_ablation_logs
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_schedules_ablation.py; sleep 15; done
mv logs dev0926-train_gpt_schedules_ablation_logs

# dev1001
source ~/dev1001/bin/activate
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt.py; sleep 15; done
mv logs dev1001-train_gpt_logs
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_pr146.py; sleep 15; done
mv logs dev1001-train_gpt_pr146_logs

# dev1031
source ~/dev1031/bin/activate
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt.py; sleep 15; done
mv logs dev1031-train_gpt_logs
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_pr146.py; sleep 15; done
mv logs dev1031-train_gpt_pr146_logs
