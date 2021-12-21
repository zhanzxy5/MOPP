#!/usr/bin/env bash
env_evaluate=True
DATA_SOURCE='mujoco'
L2_WEIGHT=1e-5
TRAIN_STEPS=500000
N_TRAIN=-1
for SAMPLE in 100; do
n_eval_episodes=20
for penalty_lambda in 0; do
for SEED in 11 22 33 44 55; do
use_value_fn=True

train_b_alpha_entropy=False
train_d_alpha_entropy=False
b_alpha_entropy=0
d_alpha_entropy=0
q_net_ckpt_name='Q_nets/q_nets'
testing_mode=False
task=0
ENV='Hopper-v2'
TASK='hopper'
for TYPE in 'random' 'medium' 'mixed' 'medium_expert_v1_old'; do
case $TYPE in
'random')
KAPPA=10
H=4
SIGMA=0.45
BETA=0
d_uncertainty_threshold=0.5
# b_list='(0,1,2)'
b_list='(2,3,5)'
d_list='(1,2,4)'
;;
'medium')
KAPPA=0.3
H=4
SIGMA=0.25
BETA=0
d_uncertainty_threshold=1
# b_list='(0,4,5)'
b_list='(2,3,4)'
d_list='(0,3,4)'
;;
'mixed')
KAPPA=0.3
H=4
SIGMA=0.6
BETA=0
d_uncertainty_threshold=1
# b_list='(1,2,4)'
b_list='(1,3,5)'
d_list='(0,3,4)'
;;
'medium_expert')
KAPPA=3
H=10
# SIGMA=0.03
BETA=0
# d_uncertainty_threshold=1
b_list='(2,3,5)'
d_list='(0,3,5)'
;;
'medium_expert_v1_old')
KAPPA=3
H=10
SIGMA=0.4
BETA=0
d_uncertainty_threshold=1
# b_list='(0,4,5)'
b_list='(0,3,4)'
d_list='(0,2,3)'
;;
esac
DATA=$TASK'_'$TYPE
maxq=True
test_b_only=False
for model_id in 0; do
GPU_DEVICE=0
AGENT_NAME='mopp'
AGENT='mopp_agent'
b_list=None
# offline mppi
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python train_offline.py \
    --alsologtostderr --sub_dir='test/k'$KAPPA'-H'$H'-S'$SAMPLE'-Sig'$SIGMA \
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
    --b_ckpt='behavior/b_adm' \
    --f_ckpt='dynamics/d_adm' \
    --gin_bindings="$AGENT.Agent.pred_len=$H" \
    --gin_bindings="$AGENT.Agent.pop_size=$SAMPLE" \
    --gin_bindings="$AGENT.Agent.kappa=$KAPPA" \
    --gin_bindings="$AGENT.Agent.noise_sigma=$SIGMA" \
    --gin_bindings="$AGENT.Agent.beta=$BETA" \
    --gin_bindings="$AGENT.Agent.train_b_alpha_entropy=$train_b_alpha_entropy" \
    --gin_bindings="$AGENT.Agent.b_alpha_entropy=$b_alpha_entropy" \
    --gin_bindings="$AGENT.Agent.train_d_alpha_entropy=$train_d_alpha_entropy" \
    --gin_bindings="$AGENT.Agent.d_alpha_entropy=$d_alpha_entropy" \
    --gin_bindings="$AGENT.Agent.d_uncertainty_threshold=$d_uncertainty_threshold" \
    --gin_bindings="$AGENT.Agent.penalty_lambda=$penalty_lambda" \
    --gin_bindings="$AGENT.Agent.use_value_fn=$use_value_fn" \
    --gin_bindings="$AGENT.Agent.b_list=$b_list" \
    --gin_bindings="$AGENT.Agent.d_list=$d_list" \
    --gin_bindings="$AGENT.Agent.maxq=$maxq" \
    --gin_bindings="$AGENT.Agent.model_id=$model_id" \
    --gin_bindings="$AGENT.Agent.test_b_only=$test_b_only" \
    --gin_bindings="train_eval_planning.batch_size=512" \
    --gin_bindings="train_eval_planning.seed=$SEED" \
    --gin_bindings="train_eval_planning.weight_decays=[$L2_WEIGHT]" \
    --gin_bindings="train_eval_planning.q_net_ckpt_name='$q_net_ckpt_name'" \
    --gin_bindings="train_eval_planning.testing_mode=$testing_mode" \
    --gin_bindings="train_eval_planning.model_params=((((500,), (200, 100)), ((500,), (200, 100)), ((500, 500), 2)), ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)),
    ((0,1,2,3,4,5,6,7,8,9,10,11),(10,9,8,7,6,5,4,3,2,1,0,11),(5,4,3,2,1,0,10,9,8,7,6,11),(0,7,9,6,5,1,3,8,10,2,4,11),(2,9,6,8,7,5,10,4,1,0,3,11),(10,2,0,8,4,5,9,3,6,7,1,11)))"

sleep 2
done
done
done
done
done
