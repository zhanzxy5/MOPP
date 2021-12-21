#!/usr/bin/env bash
env_evaluate=True
L2_WEIGHT=1e-5
TRAIN_STEPS=150000
N_TRAIN=-1
DATA_SOURCE='adroit'
for DATA_ in 'expert-v1'; do
for TASK in 'relocate'; do

for SEED in 11 22 33 44 55; do
for KAPPA in 0.3 0.03; do
for SIGMA in 0.7 0.8 0.9 1.0 1.1; do
for uncer_per in 85; do
for SAMPLE in 100; do
for H in 4; do
n_eval_episodes=20
use_value_fn=True
q_net_ckpt_name='Q_nets/B512/q_nets'
testing_mode=False
b_list=None
d_list=None
ENV=$TASK'-v0'
DATA=$TASK'-'$DATA_
maxq=True
test_b_only=False
for model_id in 0; do
d_uncertainty_threshold=None
BETA=0
GPU_DEVICE=0
AGENT_NAME='mopp'
AGENT='mopp_agent'
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
    --b_ckpt='behavior/B512/b_adm' \
    --f_ckpt='dynamics/B512/d_adm' \
    --gin_bindings="$AGENT.Agent.pred_len=$H" \
    --gin_bindings="$AGENT.Agent.pop_size=$SAMPLE" \
    --gin_bindings="$AGENT.Agent.kappa=$KAPPA" \
    --gin_bindings="$AGENT.Agent.noise_sigma=$SIGMA" \
    --gin_bindings="$AGENT.Agent.beta=$BETA" \
    --gin_bindings="$AGENT.Agent.d_uncertainty_threshold=$d_uncertainty_threshold" \
    --gin_bindings="$AGENT.Agent.use_value_fn=$use_value_fn" \
    --gin_bindings="$AGENT.Agent.b_list=$b_list" \
    --gin_bindings="$AGENT.Agent.d_list=$d_list" \
    --gin_bindings="$AGENT.Agent.maxq=$maxq" \
    --gin_bindings="$AGENT.Agent.model_id=$model_id" \
    --gin_bindings="$AGENT.Agent.test_b_only=$test_b_only" \
    --gin_bindings="$AGENT.Agent.uncertainty_percentile=$uncer_per" \
    --gin_bindings="train_eval_planning.batch_size=512" \
    --gin_bindings="train_eval_planning.seed=$SEED" \
    --gin_bindings="train_eval_planning.weight_decays=[$L2_WEIGHT]" \
    --gin_bindings="train_eval_planning.q_net_ckpt_name='$q_net_ckpt_name'" \
    --gin_bindings="train_eval_planning.testing_mode=$testing_mode" \
    --gin_bindings="train_eval_planning.model_params=((((500,), (200, 100)), ((500,), (200, 100)), ((500, 500), 2)), None, None)"

sleep 2
done
done
done
done
done
done
done
done
done