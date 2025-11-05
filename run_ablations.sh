#!/bin/bash

# stable29
echo "========================================="
echo "Starting stable29 environment"
echo "========================================="
source ~/stable29/bin/activate
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

echo "Running train_gpt.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt.py; sleep 15; done
mv logs stable29-train_gpt_logs
echo "Completed train_gpt.py, logs moved to stable29-train_gpt_logs"
echo ""

echo "Running train_gpt_pr146.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_pr146.py; sleep 15; done
mv logs stable29-train_gpt_pr146_logs
echo "Completed train_gpt_pr146.py, logs moved to stable29-train_gpt_pr146_logs"
echo ""

# dev0926 (with extra ablation scripts)
echo "========================================="
echo "Starting dev0926 environment"
echo "========================================="
source ~/dev0926/bin/activate
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

echo "Running train_gpt.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt.py; sleep 15; done
mv logs dev0926-train_gpt_logs
echo "Completed train_gpt.py, logs moved to dev0926-train_gpt_logs"
echo ""

echo "Running train_gpt_pr146.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_pr146.py; sleep 15; done
mv logs dev0926-train_gpt_pr146_logs
echo "Completed train_gpt_pr146.py, logs moved to dev0926-train_gpt_pr146_logs"
echo ""

echo "Running train_gpt_1dim_ablation.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_1dim_ablation.py; sleep 15; done
mv logs dev0926-train_gpt_1dim_ablation_logs
echo "Completed train_gpt_1dim_ablation.py, logs moved to dev0926-train_gpt_1dim_ablation_logs"
echo ""

echo "Running train_gpt_attn_reshape_ablation.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_attn_reshape_ablation.py; sleep 15; done
mv logs dev0926-train_gpt_attn_reshape_ablation_logs
echo "Completed train_gpt_attn_reshape_ablation.py, logs moved to dev0926-train_gpt_attn_reshape_ablation_logs"
echo ""

echo "Running train_gpt_lr_mul_ablation.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_lr_mul_ablation.py; sleep 15; done
mv logs dev0926-train_gpt_lr_mul_ablation_logs
echo "Completed train_gpt_lr_mul_ablation.py, logs moved to dev0926-train_gpt_lr_mul_ablation_logs"
echo ""

echo "Running train_gpt_schedules_ablation.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_schedules_ablation.py; sleep 15; done
mv logs dev0926-train_gpt_schedules_ablation_logs
echo "Completed train_gpt_schedules_ablation.py, logs moved to dev0926-train_gpt_schedules_ablation_logs"
echo ""

# dev1001
echo "========================================="
echo "Starting dev1001 environment"
echo "========================================="
source ~/dev1001/bin/activate
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

echo "Running train_gpt.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt.py; sleep 15; done
mv logs dev1001-train_gpt_logs
echo "Completed train_gpt.py, logs moved to dev1001-train_gpt_logs"
echo ""

echo "Running train_gpt_pr146.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_pr146.py; sleep 15; done
mv logs dev1001-train_gpt_pr146_logs
echo "Completed train_gpt_pr146.py, logs moved to dev1001-train_gpt_pr146_logs"
echo ""

# dev1031
echo "========================================="
echo "Starting dev1031 environment"
echo "========================================="
source ~/dev1031/bin/activate
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

echo "Running train_gpt.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt.py; sleep 15; done
mv logs dev1031-train_gpt_logs
echo "Completed train_gpt.py, logs moved to dev1031-train_gpt_logs"
echo ""

echo "Running train_gpt_pr146.py (5 iterations)..."
for i in {1..5}; do torchrun --standalone --nproc_per_node=8 train_gpt_pr146.py; sleep 15; done
mv logs dev1031-train_gpt_pr146_logs
echo "Completed train_gpt_pr146.py, logs moved to dev1031-train_gpt_pr146_logs"
echo ""

echo "========================================="
echo "All training runs completed!"
echo "========================================="
