# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for offline RL."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import datetime
import re

import numpy as np
import random
import tensorflow as tf


def get_summary_str(step=None, info=None, prefix=''):
  summary_str = prefix
  if step is not None:
    summary_str += 'Step %d; ' % (step)
  for key, val in info.items():
    if isinstance(val, (int, np.int32, np.int64)):
      summary_str += '%s %d; ' % (key, val)
    elif isinstance(val, (float, np.float32, np.float64)):
      summary_str += '%s %.4g; ' % (key, val)
  return summary_str


def write_summary(summary_writer, step, info):
  with summary_writer.as_default():
    for key, val in info.items():
      if isinstance(
          val, (int, float, np.int32, np.int64, np.float32, np.float64)):
        tf.compat.v2.summary.scalar(name=key, data=val, step=step)


def soft_variables_update(source_variables, target_variables, tau=1.0):
  for (v_s, v_t) in zip(source_variables, target_variables):
    v_t.assign((1 - tau) * v_t + tau * v_s)


def shuffle_indices_with_steps(n, steps=1, rand=None):
  """Randomly shuffling indices while keeping segments."""
  if steps == 0:
    return np.arange(n)
  if rand is None:
    rand = np.random
  n_segments = int(n // steps)
  n_effective = n_segments * steps
  batch_indices = rand.permutation(n_segments)
  batches = np.arange(n_effective).reshape([n_segments, steps])
  shuffled_batches = batches[batch_indices]
  shuffled_indices = np.arange(n)
  shuffled_indices[:n_effective] = shuffled_batches.reshape([-1])
  return shuffled_indices


def clip_by_eps(x, spec, eps=0.0):
  return tf.clip_by_value(
      x, spec.minimum + eps, spec.maximum - eps)


def add_gaussian_noise(actions, spec, std):
  noise = spec.maximum * tf.random.normal(shape=actions.shape, stddev=std)
  return tf.clip_by_value(actions + noise, spec.minimum, spec.maximum)


def get_optimizer(name):
  """Get an optimizer generator that returns an optimizer according to lr."""
  if name == 'adam':
    def adam_opt_(lr):
      return tf.keras.optimizers.Adam(lr=lr)
    return adam_opt_
  else:
    raise ValueError('Unknown optimizer %s.' % name)


def load_variable_from_ckpt(ckpt_name, var_name):
  var_name_ = '/'.join(var_name.split('.')) + '/.ATTRIBUTES/VARIABLE_VALUE'
  return tf.train.load_variable(ckpt_name, var_name_)


def soft_relu(x):
  """Compute log(1 + exp(x))."""
  # Note: log(sigmoid(x)) = x - soft_relu(x) = - soft_relu(-x).
  #       log(1 - sigmoid(x)) = - soft_relu(x)
  # tf.math.softplus
  return tf.log(1.0 + tf.exp(-tf.abs(x))) + tf.maximum(x, 0.0)


@tf.custom_gradient
def relu_v2(x):
  """Relu with modified gradient behavior."""
  value = tf.nn.relu(x)
  def grad(dy):
    if_y_pos = tf.cast(tf.greater(dy, 0.0), tf.float32)
    if_x_pos = tf.cast(tf.greater(x, 0.0), tf.float32)
    return (if_y_pos * if_x_pos + (1.0 - if_y_pos)) * dy
  return value, grad


@tf.custom_gradient
def clip_v2(x, low, high):
  """Clipping with modified gradient behavior."""
  value = tf.minimum(tf.maximum(x, low), high)
  def grad(dy):
    if_y_pos = tf.cast(tf.greater(dy, 0.0), tf.float32)
    if_x_g_low = tf.cast(tf.greater(x, low), tf.float32)
    if_x_l_high = tf.cast(tf.less(x, high), tf.float32)
    return (if_y_pos * if_x_g_low +
            (1.0 - if_y_pos) * if_x_l_high) * dy
  return value, grad


class Flags(object):

  def __init__(self, **kwargs):
    for key, val in kwargs.items():
      setattr(self, key, val)


def get_datetime():
  now = datetime.datetime.now().isoformat()
  now = re.sub(r'\D', '', now)[:-6]
  return now


def maybe_makedirs(log_dir):
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)


def subsample_trajectories(expert_states, expert_actions, expert_next_states, expert_next_actions,
                           expert_rewards, expert_dones, num_trajectories=None, rank_identifier='reward'):
  """Extracts a (random) subset of trajectories.

  Args:
    expert_states: A numpy array with expert states.
    expert_actions: A numpy array with expert states.
    expert_next_states: A numpy array with expert states.
    expert_dones: A numpy array with expert states.
    num_trajectories: A number of trajectories to extract.

  Returns:
      Numpy arrays that contain states, actions, next_states and dones.
  """
  assert rank_identifier == 'random' or 'reward' or 'origin', 'wrong rank identifier!!!'
  expert_states_traj = [[]]
  expert_actions_traj = [[]]
  expert_next_states_traj = [[]]
  expert_next_actions_traj = [[]]
  expert_rewards_traj = [[]]
  expert_dones_traj = [[]]
  traj_sum_rewards = []

  for i in range(expert_states.shape[0]):
    expert_states_traj[-1].append(expert_states[i])
    expert_actions_traj[-1].append(expert_actions[i])
    expert_next_states_traj[-1].append(expert_next_states[i])
    expert_next_actions_traj[-1].append(expert_next_actions[i])
    expert_rewards_traj[-1].append(expert_rewards[i])
    expert_dones_traj[-1].append(expert_dones[i])

    if expert_dones[i]:
      traj_sum_rewards.append(np.sum(expert_rewards_traj[-1]))

    if expert_dones[i] and i < expert_states.shape[0] - 1:
      expert_states_traj.append([])
      expert_actions_traj.append([])
      expert_next_states_traj.append([])
      expert_next_actions_traj.append([])
      expert_rewards_traj.append([])
      expert_dones_traj.append([])

  if rank_identifier == 'random':
    shuffle_inds = list(range(len(expert_states_traj)))
    random.shuffle(shuffle_inds)
  elif rank_identifier == 'reward':
    sort_inds = np.argsort(traj_sum_rewards)[::-1]
    shuffle_inds = list(sort_inds)
  elif rank_identifier == 'origin':
    shuffle_inds = list(range(len(expert_states_traj)))

  if num_trajectories is None:
    num_trajectories = len(shuffle_inds)
  shuffle_inds = shuffle_inds[:num_trajectories]
  expert_states_traj = [expert_states_traj[i] for i in shuffle_inds]
  expert_actions_traj = [expert_actions_traj[i] for i in shuffle_inds]
  expert_next_states_traj = [expert_next_states_traj[i] for i in shuffle_inds]
  expert_next_actions_traj = [expert_next_actions_traj[i] for i in shuffle_inds]
  expert_rewards_traj = [expert_rewards_traj[i] for i in shuffle_inds]
  expert_dones_traj = [expert_dones_traj[i] for i in shuffle_inds]

  def concat_trajectories(trajectories):
    return np.concatenate(trajectories, 0)

  expert_states = concat_trajectories(expert_states_traj)
  expert_actions = concat_trajectories(expert_actions_traj)
  expert_next_states = concat_trajectories(expert_next_states_traj)
  expert_next_actions = concat_trajectories(expert_next_actions_traj)
  expert_rewards = concat_trajectories(expert_rewards_traj)
  expert_dones = concat_trajectories(expert_dones_traj)

  return expert_states, expert_actions, expert_next_states, expert_next_actions, \
         expert_rewards, expert_dones, traj_sum_rewards


def trans2trajs(expert_states, expert_actions, expert_next_states,
                expert_next_actions, expert_rewards, expert_dones, traj_len=1000):
  """Extracts trajectories from transitions.

  Args:
    expert_states: A numpy array with expert states.
    expert_actions: A numpy array with expert states.
    expert_next_states: A numpy array with expert states.
    expert_dones: A numpy array with expert states.
    num_trajectories: A number of trajectories to extract.

  Returns:
      Numpy arrays that contain states, actions, next_states and dones.
  """
  expert_states_traj = [[]]
  expert_actions_traj = [[]]
  expert_next_states_traj = [[]]
  expert_next_actions_traj = [[]]
  expert_rewards_traj = [[]]
  expert_dones_traj = [[]]
  traj_sum_rewards = []
  len = 0

  for i in range(expert_states.shape[0]):
    if len < traj_len:
      expert_states_traj[-1].append(expert_states[i])
      expert_actions_traj[-1].append(expert_actions[i])
      expert_next_states_traj[-1].append(expert_next_states[i])
      expert_next_actions_traj[-1].append(expert_next_actions[i])
      expert_rewards_traj[-1].append(expert_rewards[i])
      expert_dones_traj[-1].append(expert_dones[i])
      len += 1

    if expert_dones[i]:
      traj_sum_rewards.append(np.sum(expert_rewards_traj[-1]))

    if expert_dones[i] and i < expert_states.shape[0] - 1:
      expert_states_traj.append([])
      expert_actions_traj.append([])
      expert_next_states_traj.append([])
      expert_next_actions_traj.append([])
      expert_rewards_traj.append([])
      expert_dones_traj.append([])
      len = 0

  return expert_states_traj, expert_actions_traj, expert_next_states_traj, \
         expert_next_actions_traj, expert_rewards_traj, expert_dones_traj, traj_sum_rewards


def trajs2trans(expert_states_traj, expert_actions_traj, expert_next_states_traj,
                expert_next_actions_traj, expert_dones_traj):
  """Concat transitions back to trajectories.

  Args:
    expert_states: A numpy array with expert states.
    expert_actions: A numpy array with expert states.
    expert_next_states: A numpy array with expert states.
    expert_dones: A numpy array with expert states.
    num_trajectories: A number of trajectories to extract.

  Returns:
      Numpy arrays that contain states, actions, next_states and dones.
  """
  def concat_trajectories(trajectories):
    return np.concatenate(trajectories, 0)

  expert_states = concat_trajectories(expert_states_traj)
  expert_actions = concat_trajectories(expert_actions_traj)
  expert_next_states = concat_trajectories(expert_next_states_traj)
  expert_next_actions = concat_trajectories(expert_next_actions_traj)
  expert_dones = concat_trajectories(expert_dones_traj)

  return expert_states, expert_actions, expert_next_states, \
         expert_next_actions, expert_dones

def set_seed(seed):
  seed %= 4294967294
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)
  print("Using seed {}".format(seed))