from absl import logging

import tensorflow as tf0
import tensorflow.compat.v1 as tf
import numpy as np
import gym
import h5py

from scripts import utils
from scripts import dataset
from scripts import wrappers
from gym.wrappers import time_limit

from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper


def get_keys(h5file):
  keys = []

  def visitor(name, item):
    if isinstance(item, h5py.Dataset):
      keys.append(name)

  h5file.visititems(visitor)
  return keys


def get_data_env_d4rl(gym_env, file_name, identifier='model_free', num_transitions=-1,
                      normalize_states=False, scale_rewards=False, per=False):
  """get transitions"""
  s_dim, a_dim = gym_env

  dataset_file = h5py.File(file_name, 'r')
  # offline_dataset = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
  data_dict = {}
  for k in get_keys(dataset_file):
    try:
      # first try loading as an array
      data_dict[k] = dataset_file[k][:]
    except ValueError as e:  # try loading as a scalar
      data_dict[k] = dataset_file[k][()]
  dataset_file.close()
  offline_dataset = data_dict

  dataset_size = len(offline_dataset['observations'])
  observation_dtype = tf.float32  # gym_env.observation_space.dtype
  action_dtype = tf.float32  # gym_env.action_space.dtype

  offline_dataset['terminals'] = np.squeeze(offline_dataset['terminals'])
  offline_dataset['rewards'] = np.squeeze(offline_dataset['rewards'])

  nonterminal_steps, = np.where(
      np.logical_and(
          np.logical_not(offline_dataset['terminals']),
          np.arange(dataset_size) < dataset_size - 1))
  logging.info('Found %d non-terminal steps out of a total of %d steps.' % (
      len(nonterminal_steps), dataset_size))

  demo_s1 = offline_dataset['observations'][nonterminal_steps]
  demo_s2 = offline_dataset['observations'][nonterminal_steps + 1]
  demo_a1 = offline_dataset['actions'][nonterminal_steps]
  demo_a2 = offline_dataset['actions'][nonterminal_steps + 1]
  demo_r  = offline_dataset['rewards'][nonterminal_steps]
  demo_d  = offline_dataset['terminals'][nonterminal_steps + 1]

  if num_transitions != -1:
    demo_s1 = demo_s1[:num_transitions]
    demo_s2 = demo_s2[:num_transitions]
    demo_a1 = demo_a1[:num_transitions]
    demo_a2 = demo_a2[:num_transitions]
    demo_r  = demo_r[:num_transitions]
    demo_d  = demo_d[:num_transitions]

  # (demo_s1, demo_a1, demo_s2, demo_a2, demo_r, demo_d,
  #  traj_sum_rewards) = utils.subsample_trajectories(demo_s1,
  #                                                   demo_a1,
  #                                                   demo_s2,
  #                                                   demo_a2,
  #                                                   demo_r,
  #                                                   demo_d,
  #                                                   num_trajectories=None,
  #                                                   rank_identifier='origin')

  dataset_size = demo_s1.shape[0]
  # traj_mean_rewards = np.mean(traj_sum_rewards)
  # print('{} demonstraions, mean reward is {}'.format(dataset_size, traj_mean_rewards))

  if normalize_states:
    shift = -np.mean(demo_s1, 0)
    scale = 1.0 / (np.std(demo_s1, 0) + 1e-3)
    demo_s1 = (demo_s1 + shift) * scale
    demo_s2 = (demo_s2 + shift) * scale
  else:
    shift = None
    scale = None

  if scale_rewards:
    r_max = np.max(demo_r)
    r_min = np.min(demo_r)
    demo_r = (demo_r - r_min) / (r_max - r_min)

  if per:
    demo_dataset = dataset.PrioritizedReplayBuffer(
        s_dim,
        a_dim,
        identifier=identifier,
        size=dataset_size)
  else:
    demo_dataset = dataset.Dataset(
        s_dim,
        a_dim,
        identifier=identifier,
        size=dataset_size)

  demo_s1 = tf.convert_to_tensor(demo_s1, dtype=observation_dtype)
  demo_s2 = tf.convert_to_tensor(demo_s2, dtype=observation_dtype)
  demo_a1 = tf.convert_to_tensor(demo_a1, dtype=action_dtype)
  demo_a2 = tf.convert_to_tensor(demo_a2, dtype=action_dtype)
  demo_r  = tf.convert_to_tensor(demo_r, dtype=tf.float32)
  demo_d  = tf.convert_to_tensor(1. - demo_d, dtype=tf.float32)

  transitions = dataset.Transition_model_free(demo_s1, demo_s2, demo_a1, demo_a2, demo_d, demo_r)

  demo_dataset.add_transitions(transitions)

  # Split data.
  rand = np.random.RandomState(0)
  indices = utils.shuffle_indices_with_steps(
      n=demo_dataset.size, steps=0, rand=rand)
  train_data = demo_dataset.create_view(indices)
  print('n_train is %d' % (len(indices)))

  return train_data, shift, scale
