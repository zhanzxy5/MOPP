#!/usr/bin/env bash
env_evaluate=True
L2_WEIGHT=1e-5
TRAIN_STEPS=80000
N_TRAIN=-1
DATA_SOURCE='adroit'
for DATA_ in 'expert-v1'; do
for TASK in 'pen' 'door' 'hammer' 'relocate'; do
for KAPPA in 0.01 0.03 0.1 0.3; do
for SIGMA in 0.05 0.1 0.2; do
for SAMPLE in 100 200; do
for H in 4 8 16; do
for SEED in 11 22 33 44 55; do
n_eval_episodes=20
use_value_fn=True
q_net_ckpt_name='mbop/Q_nets/q_nets_H'$H
testing_mode=False
ENV=$TASK'-v0'
DATA=$TASK'-'$DATA_
maxq=True
test_b_only=False
model_id=0
BETA=0
GPU_DEVICE=0
AGENT_NAME='mbop'
AGENT='mbop_agent'
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
    --alsologtostderr --sub_dir='test/S'$SAMPLE'-H'$H'-K'$KAPPA'-Sig'$SIGMA \
    --env_evaluate=$env_evaluate \
    --env_name=$ENV \
    --agent_name=$AGENT_NAME \
    --data_file_source=$DATA_SOURCE \
    --data_file_name=$DATA \
    --total_train_steps=$TRAIN_STEPS \
    --num_transitions=$N_TRAIN \
    --seed=$SEED \
    --identifier='planning' \
    --n_eval_episodes=$n_eval_episodes \
    --b_ckpt='mbop/behavior/b_nets' \
    --f_ckpt='mbop/dynamics/d_nets' \
    --gin_bindings="$AGENT.Agent.pred_len=$H" \
    --gin_bindings="$AGENT.Agent.pop_size=$SAMPLE" \
    --gin_bindings="$AGENT.Agent.kappa=$KAPPA" \
    --gin_bindings="$AGENT.Agent.noise_sigma=$SIGMA" \
    --gin_bindings="$AGENT.Agent.beta=$BETA" \
    --gin_bindings="$AGENT.Agent.use_value_fn=$use_value_fn" \
    --gin_bindings="$AGENT.Agent.model_id=$model_id" \
    --gin_bindings="$AGENT.Agent.test_b_only=$test_b_only" \
    --gin_bindings="train_eval_planning.batch_size=512" \
    --gin_bindings="train_eval_planning.seed=$SEED" \
    --gin_bindings="train_eval_planning.weight_decays=[$L2_WEIGHT]" \
    --gin_bindings="train_eval_planning.q_net_ckpt_name='$q_net_ckpt_name'" \
    --gin_bindings="train_eval_planning.testing_mode=$testing_mode" \
    --gin_bindings="train_eval_planning.model_params=((((500, 500), 3), ((500, 500), 3), ((500, 500), 3)), None, None)"

sleep 2
done
done
done
done
done
done
done