"""
Training and evaluation in the offline mode.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

from absl import logging

import gin
import gym
import numpy as np
import tensorflow as tf0
import tensorflow.compat.v1 as tf
import importlib
import sys
from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper

from scripts import dataset
from scripts import train_eval_utils
from scripts import utils
import get_offline_data


@gin.configurable
def train_eval_planning(
        # Basic args.
        log_dir,
        benchmark_name,
        model_dir,
        data_file_source,
        data_file,
        agent_module,
        env_name='HalfCheetah-v2',
        seed=0,
        # Train and eval args.
        total_train_steps=int(5e4),
        summary_freq=100,
        print_freq=1000,
        save_freq=int(2e4),
        eval_freq=5000,
        n_eval_episodes=20,
        # Agent args.
        model_params=((((300, 300), (300, 100)), ((300, 300), (300, 100))), None, None),
        optimizers=(('adam', 0.001),),
        batch_size=256,
        weight_decays=(0.0,),
        update_freq=1,
        update_rate=0.005,
        discount=0.99,
        # num_trajectories=None,  # how many ep trajectories to use
        num_transitions=None,  # how many ep transitions to use
        normalize_states=False,  # whether normalize states.
        behavior_ckpt_name=None,
        dynamics_ckpt_name=None,
        q_net_ckpt_name='placeholder',
        testing_mode=False,
        model_id=0,
        score_normalize=False,
        state_normalize=False,
        reward_normalize=False,
        evaluate=False,
        per=False,
):
    """Training a policy with a fixed dataset."""
    # Create tf_env to get specs.
    print('[train_eval_offline.py] env_name=', env_name)
    print('[train_eval_offline.py] data_file=', data_file)
    print('[train_eval_offline.py] agent_module=', agent_module)
    print('[train_eval_offline.py] model_params=', model_params)
    print('[train_eval_offline.py] optimizers=', optimizers)
    print('[train_eval_offline.py] bckpt_file=', behavior_ckpt_name)
    print('[train_eval_offline.py] fckpt_file=', dynamics_ckpt_name)
    if q_net_ckpt_name is not None:
        print('[train_eval_offline.py] q_net_ckpt_name=', q_net_ckpt_name)
    if env_name == 'Hopper-v2':
        s_dim = 11
        s_max = np.inf
        a_dim = 3
        a_max = 1
    elif env_name == 'HalfCheetah-v2':
        s_dim = 17
        s_max = np.inf
        a_dim = 6
        a_max = 1
    elif env_name == 'Walker2d-v2':
        s_dim = 17
        s_max = np.inf
        a_dim = 6
        a_max = 1
    elif 'pen' in env_name:
        s_dim = 45
        a_dim = 24
        a_max = 1
    elif 'hammer' in env_name:
        s_dim = 46
        a_dim = 26
        a_max = 1
    elif 'door' in env_name:
        s_dim = 39
        a_dim = 28
        a_max = 1
    elif 'relocate' in env_name:
        s_dim = 39
        a_dim = 30
        a_max = 1
    observation_spec = utils.Flags(shape=(s_dim,), minimum=-np.inf, maximum=np.inf)
    action_spec = utils.Flags(shape=(a_dim,), minimum=-1, maximum=a_max)
    # Prepare offline data.
    if benchmark_name == 'd4rl':
        d4rl_data_file = data_file + '.hdf5'
        data, shift, scale = get_offline_data.get_data_env_d4rl(
            [s_dim, a_dim], file_name=d4rl_data_file, identifier='model_free',
            num_transitions=num_transitions, normalize_states=state_normalize,
            scale_rewards=reward_normalize, per=per)
    else:
        raise ValueError("Wrong data file source or not supported currently!")

    # env
    if evaluate:
        import d4rl
        from tf_agents.environments import tf_py_environment
        from tf_agents.environments import gym_wrapper
        from scripts import wrappers
        gym_env = gym.make(env_name)
        gym_env.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        wrapped_env = wrappers.wrapped_il_env(gym_env, shift=shift, scale=scale)
        spec_dtype_map = {gym.spaces.Box: np.float32}  # map to float32
        tf_env = tf_py_environment.TFPyEnvironment(
            gym_wrapper.GymWrapper(wrapped_env, spec_dtype_map=spec_dtype_map))

    norm_min = -np.inf
    norm_max = np.inf
    if score_normalize:
        sys.path.append('../')
        import_path = 'data' + '.' + benchmark_name + '.' + data_file_source
        module = importlib.import_module(import_path)
        domain = env_name.split('-')[0]
        norm_min = getattr(module, (domain + '_random_score').upper())
        norm_max = getattr(module, (domain + '_expert_score').upper())

    # Create agent.
    agent_flags = utils.Flags(
        observation_spec=observation_spec,
        action_spec=action_spec,
        model_params=model_params,
        optimizers=optimizers,
        batch_size=batch_size,
        weight_decays=weight_decays,
        update_freq=update_freq,
        update_rate=update_rate,
        discount=discount,
        train_data=data,
    )
    agent_args = agent_module.Config(agent_flags).agent_args
    my_agent_arg_dict = {}
    for k in vars(agent_args):
        my_agent_arg_dict[k] = vars(agent_args)[k]
    my_agent_arg_dict['observation_spec'] = observation_spec
    print('agent_args:', my_agent_arg_dict)
    agent = agent_module.Agent(**my_agent_arg_dict)
    # agent_ckpt_name = os.path.join(log_dir, 'agent')

    # Train and restore dynamics model.
    # if dynamics_ckpt_name == 'placeholder':
    #     raise ValueError("You must train a dynamic model in model-based RL")
    dynamics_ckpt_dir = os.path.join(model_dir, dynamics_ckpt_name)
    train_d_summary_dir = os.path.join(model_dir, dynamics_ckpt_name + '_train_log')
    if dynamics_ckpt_name != 'placeholder':
        if tf.io.gfile.exists('{}.index'.format(dynamics_ckpt_dir)):
            logging.info('Checkpoint found at %s.', dynamics_ckpt_dir)
        else:
            logging.info('No trained checkpoint, train the %s.', dynamics_ckpt_name)
            train_d_summary_writer = tf0.compat.v2.summary.create_file_writer(
                train_d_summary_dir)
            for i in range(total_train_steps):
                train_f_info = agent.train_dynamics_step()
                if i % print_freq == 0:
                    logging.info(utils.get_summary_str(step=i, info=train_f_info))
                if i % summary_freq == 0 or i == total_train_steps:
                    agent.write_d_train_summary(train_d_summary_writer, i, train_f_info)
            agent.save_dynamics_model(dynamics_ckpt_dir)
        agent.restore_dynamics_model(dynamics_ckpt_dir)

    # Train behavior model if needed.
    behavior_ckpt_dir = os.path.join(model_dir, behavior_ckpt_name)
    train_b_summary_dir = os.path.join(model_dir, behavior_ckpt_name + '_train_log')
    if behavior_ckpt_name != 'placeholder':
        if tf.io.gfile.exists('{}.index'.format(behavior_ckpt_dir)):
            logging.info('Checkpoint found at %s.', behavior_ckpt_dir)
        else:
            logging.info('No trained checkpoint, train the %s.', behavior_ckpt_name)
            train_b_summary_writer = tf0.compat.v2.summary.create_file_writer(
                train_b_summary_dir)
            for i in range(total_train_steps):
                train_b_info = agent.train_behavior_step()
                if i % print_freq == 0:
                    logging.info(utils.get_summary_str(step=i, info=train_b_info))
                if i % summary_freq == 0 or i == total_train_steps:
                    agent.write_b_train_summary(train_b_summary_writer, i, train_b_info)
            agent.save_behavior_model(behavior_ckpt_dir)

    # Restore behavior model if needed.
    if behavior_ckpt_name != 'placeholder':
        agent.restore_behavior_model(behavior_ckpt_dir)

    # dynamics_uncertainty_analysis
    # for d_name in ['diff', ]:
    #     agent._dynamics_uncertainty_analysis(d_name, train_d_summary_dir)
    # return

    if testing_mode:
        a_mse = agent.test_behavior_all_data()
        s_mse, r_mse = agent.test_dynamic_all_data()
        with tf.io.gfile.GFile(train_b_summary_dir + f'/behavior{model_id}_test_results.npy', 'w') as f:
            np.save(f, np.array([a_mse]))
        with tf.io.gfile.GFile(train_d_summary_dir + f'/dynamics{model_id}_test_results.npy', 'w') as f:
            np.save(f, np.array([s_mse, r_mse]))
        return

    # train Q-net
    if q_net_ckpt_name != 'placeholder':
        q_net_ckpt_dir = os.path.join(model_dir, q_net_ckpt_name)
        if tf.io.gfile.exists('{}.index'.format(q_net_ckpt_dir)):
            logging.info('Checkpoint found at %s.', q_net_ckpt_dir)
        else:
            logging.info('No trained checkpoint, train the %s.', q_net_ckpt_name)
            train_q_summary_dir = os.path.join(model_dir, q_net_ckpt_name + '_train_log')
            train_q_summary_writer = tf0.compat.v2.summary.create_file_writer(
                train_q_summary_dir)
            for i in range(total_train_steps):
                train_q_info = agent.train_q_step(i)
                if i % print_freq == 0:
                    logging.info(utils.get_summary_str(step=i, info=train_q_info))
                if i % summary_freq == 0 or i == total_train_steps:
                    utils.write_summary(train_q_summary_writer, i, train_q_info)
            agent.save_q_model(q_net_ckpt_dir)
        agent.restore_q_model(q_net_ckpt_dir)

    # Evaluating agent.
    if evaluate:
        eval_summary_dir = os.path.join(log_dir, 'eval')
        eval_summary_writers = collections.OrderedDict()
        for policy_key in agent.test_policies.keys():
            eval_summary_writer = tf0.compat.v2.summary.create_file_writer(
                os.path.join(eval_summary_dir, policy_key))
            eval_summary_writers[policy_key] = eval_summary_writer
        eval_r_results = []
        time_st_total = time.time()
        eval_r_result, eval_r_infos = train_eval_utils.eval_policies(
            tf_env, agent.test_policies, n_eval_episodes,
            score_normalize, norm_min, norm_max)
        step = 0
        eval_r_results.append([step] + eval_r_result)
        for policy_key, policy_info in eval_r_infos.items():
            logging.info(utils.get_summary_str(
                step=None, info=policy_info, prefix=policy_key + ': '))
            utils.write_summary(eval_summary_writers[policy_key], step, policy_info)
        logging.info('Testing at step %d:', step)

        time_cost = time.time() - time_st_total
        logging.info('Evaluating finished, time cost %.4gs.', time_cost)
        return np.array(eval_r_results)
