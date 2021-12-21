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

"""Dataset for offline RL (or replay buffer for online RL)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

# from scripts.segment import MinSegmentTree, SumSegmentTree

Transition_model_free = collections.namedtuple(
    'Transition_model_free', 's1, s2, a1, a2, discount, reward')

Transition_model_based = collections.namedtuple(
    'Transition_model_based', 's1, s2, a1, a2, discount, reward, rollout_step')


class DatasetView(object):
  """Interface for reading from dataset."""

  def __init__(self, dataset, indices):
    self._dataset = dataset
    self._indices = indices

  def get_batch(self, indices):
    real_indices = self._indices[indices]
    return self._dataset.get_batch(real_indices)

  def get_all_data(self,):
    return self._dataset.data

  @property
  def size(self):
    return self._indices.shape[0]

  def update_priorities(self, inds, new_priorities):
    return self._dataset.update_priorities(inds, new_priorities)


# def save_copy(data, ckpt_name):
#   """Creates a copy of the current data and save as a checkpoint."""
#   new_data = Dataset(
#       observation_spec=data.config['observation_spec'],
#       action_spec=data.config['action_spec'],
#       size=data.size,
#       circular=False)
#   full_batch = data.get_batch(np.arange(data.size))
#   new_data.add_transitions(full_batch)
#   data_ckpt = tf.train.Checkpoint(data=new_data)
#   data_ckpt.write(ckpt_name)


class Dataset(tf.Module):
  """Tensorflow module of dataset of transitions."""

  def __init__(
      self,
      s_dim,
      a_dim,
      size,
      identifier,
      circular=True,
      ):
    super(Dataset, self).__init__()
    self._size = size
    self._circular = circular
    self._identifier = identifier
    obs_type = tf.float32  # observation_spec.dtype
    action_type = tf.float32  # action_spec.dtype
    self._s1 = self._zeros([size] + [s_dim], obs_type)
    self._s2 = self._zeros([size] + [s_dim], obs_type)
    self._a1 = self._zeros([size] + [a_dim], action_type)
    self._a2 = self._zeros([size] + [a_dim], action_type)
    self._discount = self._zeros([size], tf.float32)
    self._reward = self._zeros([size], tf.float32)
    self._a1_prob = self._zeros([size], tf.float32)
    self._rollout_step = self._zeros([size], tf.float32)

    self._data = Transition_model_free(
      s1=self._s1, s2=self._s2, a1=self._a1, a2=self._a2,
      discount=self._discount, reward=self._reward)

    self._current_size = tf.Variable(0)
    self._current_idx = tf.Variable(0)
    self._capacity = tf.Variable(self._size)
    self._next_idx = 0.0
    self._np_size = size
    # self._R = tf.Variable(0)
    # self._config = collections.OrderedDict(
    #     observation_spec=observation_spec,
    #     action_spec=action_spec,
    #     size=size,
    #     circular=circular,
    #     identifier=identifier)

  # @property
  # def config(self):
  #   return self._config

  def create_view(self, indices):
    return DatasetView(self, indices)

  def get_batch(self, indices):
    indices = tf.constant(indices)
    def get_batch_(data_):
      return tf.gather(data_, indices)
    transition_batch = tf.nest.map_structure(get_batch_, self._data)
    return transition_batch

  @property
  def data(self):
    return self._data

  @property
  def capacity(self):
    return self._size

  @property
  def size(self):
    return self._current_size.numpy()

  def _zeros(self, shape, dtype):
    """Create a variable initialized with zeros."""
    return tf.Variable(tf.zeros(shape, dtype))

  @tf.function
  def add_transitions(self, transitions):
    batch_size = transitions.s1.shape[0]
    effective_batch_size = tf.minimum(
        batch_size, self._size - self._current_idx)
    indices = self._current_idx + tf.range(effective_batch_size)
    for key in transitions._asdict().keys():
      data = getattr(self._data, key)
      batch = getattr(transitions, key)
      tf.scatter_update(data, indices, batch[:effective_batch_size])
    # Update size and index.
    if tf.less(self._current_size, self._size):
      self._current_size.assign_add(effective_batch_size)
    self._current_idx.assign_add(effective_batch_size)
    if self._circular:
      if tf.greater_equal(self._current_idx, self._size):
        self._current_idx.assign(0)
    self._R = np.arange(self._next_idx, self._next_idx + transitions.s1.shape[0])% self._np_size
    self._R = self._R.astype('int64')
    self._next_idx = (self._next_idx + transitions.s1.shape[0]) % self._np_size

class PrioritizedReplayBuffer(Dataset):

  def __init__(
          self, state_shape, action_shape, size, identifier, state_dtype=float, alpha=0.6, beta=1.0
  ):
    super(PrioritizedReplayBuffer, self).__init__(
      state_shape, action_shape, size, identifier
    )
    assert alpha >= 0
    self.alpha = alpha
    self.beta = beta

    it_capacity = 1
    while it_capacity < size:
      it_capacity *= 2

    self._it_sum = SumSegmentTree(it_capacity)
    self._it_min = MinSegmentTree(it_capacity)
    self._max_priority = 1.0

  def add_transitions(self, transitions, priorities=None):
    super().add_transitions(transitions)
    if priorities is None:
      priorities = self._max_priority
    self._it_sum[self._R] = priorities ** self.alpha
    self._it_min[self._R] = priorities ** self.alpha

  def _sample_proportional(self, batch_size):
    total = self._it_sum.sum(0, self._data.s1.shape[0] - 1)
    mass = np.random.random(size=batch_size) * total
    idx = self._it_sum.find_prefixsum_idx(mass)
    return idx

  def _sample_batch(self, indices):
    batch_size = len(indices)
    idxes = self._sample_proportional(batch_size)
    p_min = self._it_min.min() / self._it_sum.sum()
    max_weight = (p_min * len(self._data)) ** (-self.beta)
    p_sample = self._it_sum[idxes] / self._it_sum.sum()
    weights = (p_sample * len(self._data)) ** (-self.beta) / max_weight
    actor_batch = super().get_batch(idxes)
    critic_batch = super().get_batch(indices)
    return [[critic_batch, actor_batch], tf.constant(weights, tf.float32), [indices, idxes]]

  def get_batch(self, indices):
    return self._sample_batch(indices)

  def sample_uniform(self, indices):
    return super().get_batch(indices)

  def update_priorities(self, idxes, priorities):
    import tensorflow.compat.v1 as tf
    tf.config.experimental_run_functions_eagerly(True)
    priorities = priorities.numpy()
    assert len(idxes) == len(priorities)
    assert np.min(priorities) > 0
    assert np.min(idxes) >= 0
    assert np.max(idxes) < self._data.s1.shape[0]
    self._it_sum[idxes] = priorities ** self.alpha
    self._it_min[idxes] = priorities ** self.alpha
    self._max_priority = max(self._max_priority, np.max(priorities))

