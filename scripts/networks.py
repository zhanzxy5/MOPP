# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Neural network models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from typing import Optional, Union

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from scripts.distributions import DiscreteValuedDistribution

LOG_STD_MIN = -5
LOG_STD_MAX = 0


def get_spec_means_mags(spec):
  means = (spec.maximum + spec.minimum) / 2.0
  mags = (spec.maximum - spec.minimum) / 2.0
  means = tf.constant(means, dtype=tf.float32)
  mags = tf.constant(mags, dtype=tf.float32)
  return means, mags


class TD3ActorNetwork(tf.Module):
  """Deterministic Actor network."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      ):
    super(TD3ActorNetwork, self).__init__()
    self._action_spec = action_spec
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        self._action_spec.shape[0],
        activation=None,
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
    return self._action_spec

  def _get_outputs(self, state):
    h = state
    for l in self._layers:
      h = l(h)
    a_tanh = tf.tanh(h) * self._action_mags + self._action_means
    return a_tanh

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state):
    #state = tf.cast(state, dtype=tf.float64)
    return self._get_outputs(state)


class ActorNetwork(tf.Module):
  """Stochastic Actor network."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      ):
    super(ActorNetwork, self).__init__()
    self._action_spec = action_spec
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        self._action_spec.shape[0] * 2,
        activation=None,
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
    return self._action_spec

  def _get_outputs(self, state):
    h = state
    for l in self._layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    a_tanh_mode = tf.tanh(mean) * self._action_mags + self._action_means
    log_std = tf.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    a_distribution = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0.0, scale=1.0),
        bijector=tfp.bijectors.Chain([
            tfp.bijectors.AffineScalar(shift=self._action_means,
                                       scale=self._action_mags),
            tfp.bijectors.Tanh(),
            tfp.bijectors.AffineScalar(shift=mean, scale=std),
        ]),
        event_shape=[mean.shape[-1]],
        batch_shape=[mean.shape[0]])
    return a_distribution, a_tanh_mode

  def get_log_density(self, state, action):
    a_dist, _ = self._get_outputs(state)
    log_density = a_dist.log_prob(action)
    return log_density

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state):
    #state = tf.cast(state, dtype=tf.float64)
    a_dist, a_tanh_mode = self._get_outputs(state)
    a_sample = a_dist.sample()
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def call_dist(self, state):
    a_dist, a_tanh_mode = self._get_outputs(state)
    a_sample = a_dist.sample()
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a, a_dist

  def sample_n(self, state, n=1):
    a_dist, a_tanh_mode = self._get_outputs(state)
    a_sample = a_dist.sample(n)
    log_pi_a = a_dist.log_prob(a_sample)
    return a_tanh_mode, a_sample, log_pi_a

  def sample(self, state):
    return self.sample_n(state, n=1)[1][0]


class CriticNetwork(tf.Module):
  """Critic Network."""

  def __init__(
      self,
      fc_layer_params=(),
      ):
    super(CriticNetwork, self).__init__()
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, state, action):
    # state = tf.cast(state, dtype=tf.float64)
    # action = tf.cast(action, dtype=tf.float64)
    h = tf.concat([state, action], axis=-1)
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class EnsembleCriticNetwork(tf.Module):
  """Ensemble Critic Network."""

  def __init__(
      self,
      fc_layer_params=(),
      num_heads=2,
      ):
    super(EnsembleCriticNetwork, self).__init__()
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        num_heads,
        activation=None,
        )
    self._layers.append(output_layer)
    self._num_heads = num_heads

  def __call__(self, state, action):
    # state = tf.cast(state, dtype=tf.float64)
    # action = tf.cast(action, dtype=tf.float64)
    h = tf.concat([state, action], axis=-1)
    for l in self._layers:
      h = l(h)
    return h

  def get_q(self, state, action):
    h = tf.concat([state, action], axis=-1)
    for l in self._layers:
      h = l(h)
    return tf.reduce_mean(h, -1)

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class ValueNetwork(tf.Module):
  """Value Network."""

  def __init__(
      self,
      fc_layer_params=(),
      ):
    super(ValueNetwork, self).__init__()
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, state):
    # state = tf.cast(state, dtype=tf.float64)
    h = state
    for l in self._layers:
      h = l(h)
    return tf.reshape(h, [-1])

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class DiscreteValuedCriticNetwork(tf.Module):
  """Represents a Parameterized Discrete Valued Distributional Critic Network."""

  def __init__(
      self,
      vmin: Union[float, np.ndarray, tf.Tensor],
      vmax: Union[float, np.ndarray, tf.Tensor],
      num_atoms: int,
      fc_layer_params=(),
      ):
    super(DiscreteValuedCriticNetwork, self).__init__()
    vmin = tf.convert_to_tensor(vmin)
    vmax = tf.convert_to_tensor(vmax)
    self._values = tf.linspace(vmin, vmax, num_atoms)
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        tf.size(self._values),
        activation=None,
        )
    self._layers.append(output_layer)

  def __call__(self, state, action):
    # state = tf.cast(state, dtype=tf.float64)
    # action = tf.cast(action, dtype=tf.float64)
    h = tf.concat([state, action], axis=-1)
    for l in self._layers:
      h = l(h)
    return DiscreteValuedDistribution(self._values, logits=h)

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list


class BCQActorNetwork(tf.Module):
  """Actor network for BCQ."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      ):
    super(BCQActorNetwork, self).__init__()
    self._action_spec = action_spec
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
        self._action_spec.shape[0],
        activation=None,
        )
    self._layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)

  @property
  def action_spec(self):
    return self._action_spec

  def _get_outputs(self, state, action, max_perturbation):
    h = tf.concat([state, action], axis=-1)
    for l in self._layers:
      h = l(h)
    a = tf.tanh(h) * self._action_mags * max_perturbation + action
    a = tf.clip_by_value(
        a, self._action_spec.minimum, self._action_spec.maximum)
    return a

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state, action, max_perturbation):
    return self._get_outputs(state, action, max_perturbation)


class ConditionalVAENetwork(tf.Module):
  """VAE for learned behavior policy used by BCQ."""

  def __init__(
      self,
      action_spec,
      fc_layer_params=(),
      latent_dim=None,
      ):
    super(ConditionalVAENetwork, self).__init__()
    if latent_dim is None:
      latent_dim = action_spec.shape[0] * 2
    self._action_spec = action_spec
    self._encoder_layers = []

    relu_gain = tf.math.sqrt(2.0)
    relu_orthogonal = tf.keras.initializers.Orthogonal(relu_gain)
    near_zero_orthogonal = tf.keras.initializers.Orthogonal(1e-2)

    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          kernel_initializer=relu_orthogonal,
      )
      self._encoder_layers.append(l)
    output_layer = tf.keras.layers.Dense(
        latent_dim * 2,
        activation=None,
        kernel_initializer=near_zero_orthogonal,
    )
    self._encoder_layers.append(output_layer)
    self._decoder_layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          kernel_initializer=relu_orthogonal,
      )
      self._decoder_layers.append(l)
    output_layer = tf.keras.layers.Dense(
        action_spec.shape[0],
        activation=None,
        kernel_initializer=near_zero_orthogonal,
    )
    self._decoder_layers.append(output_layer)
    self._action_means, self._action_mags = get_spec_means_mags(
        self._action_spec)
    self._latent_dim = latent_dim

  @property
  def action_spec(self):
    return self._action_spec

  def forward(self, state, action):
    h = tf.concat([state, action], axis=-1)
    for l in self._encoder_layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    # std = tf.exp(tf.clip_by_value(log_std, -4, 15))
    log_std = tf.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)

    z = mean + std * tf.random.normal(shape=std.shape)
    a = self.decode(state, z)
    return a, mean, std

  def get_latent_kl(self, state, action):
    h = tf.concat([state, action], axis=-1)
    for l in self._encoder_layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    # std = tf.exp(tf.clip_by_value(log_std, -4, 15))
    std = tf.exp(log_std)
    kl = -0.5 * (1.0 + tf.log(tf.square(std)) - tf.square(mean) - tf.square(std))
    return kl

  def decode(self, state, z):
    h = tf.concat([state, z], axis=-1)
    for l in self._decoder_layers:
      h = l(h)
    a = tf.tanh(h) * self._action_mags + self._action_means
    # a = tf.clip_by_value(
    #     a, self._action_spec.minimum, self._action_spec.maximum)
    return a

  def decode_multiple(self, state, z=None, n=10):
    if z is None:
      z = tf.random.normal(shape=(n, state.shape[0], self._latent_dim))
      z = tf.clip_by_value(z, -0.5, 0.5)
    # state_tile = tf.reshape(tf.tile(state, (1, n)), (state.shape[0], n, -1))
    state_tile = tf.reshape(tf.tile(state, (n, 1)), (n, state.shape[0], -1))
    h = tf.concat([state_tile, z], axis=-1)
    for l in self._decoder_layers:
      h = l(h)
    a = tf.tanh(h) * self._action_mags + self._action_means
    # a = tf.clip_by_value(
    #     a, self._action_spec.minimum, self._action_spec.maximum)
    return a

  def sample(self, state):
    z = tf.random.normal(shape=[state.shape[0], self._latent_dim])
    z = tf.clip_by_value(z, -0.5, 0.5)
    return self.decode(state, z)

  def sample_n(self, state, n=10):
    # used in computing divergence
    a_n = self.decode_multiple(state, n=n)
    return None, a_n, None

  def get_log_density(self, state, action):
    # variational lower bound
    a_recon, mean, std = self.forward(state, action)
    # log_2pi = tf.log(tf.constant(math.pi))
    # recon = - 0.5 * tf.reduce_mean(
    #     tf.square(a_recon - action) + log_2pi, axis=-1)
    # kl = 0.5 * tf.reduce_mean(
    #     - 1.0 - tf.log(tf.square(std)) + tf.square(mean) + tf.square(std),
    #     axis=-1)
    recon_loss = tf.reduce_mean(tf.square(a_recon - action), axis=-1)
    kl_losses = -0.5 * (1.0 + tf.log(tf.square(std)) - tf.square(mean) -
                        tf.square(std))
    kl_loss = tf.reduce_mean(kl_losses, axis=-1)
    b_loss = recon_loss + kl_loss * 0.5  # Based on the pytorch implementation.
    return -b_loss

  @property
  def weights(self):
    w_list = []
    for l in self._encoder_layers:
      w_list.append(l.weights[0])
    for l in self._decoder_layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state, action):
    return self.forward(state, action)


class VAENetwork(tf.Module):
  """Vanilla VAE."""

  def __init__(
      self,
      state_spec,
      fc_layer_params,
      latent_dim=None,
      ):
    super(VAENetwork, self).__init__()
    if latent_dim is None:
      latent_dim = state_spec.shape[0] * 2
    self._state_spec = state_spec
    self._encoder_layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._encoder_layers.append(l)
    output_layer = tf.keras.layers.Dense(
        latent_dim * 2,
        activation=None)
    self._encoder_layers.append(output_layer)
    self._decoder_layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
          n_units,
          activation=tf.nn.relu,
          )
      self._decoder_layers.append(l)
    output_layer = tf.keras.layers.Dense(
        state_spec.shape[0],
        activation=None)
    self._decoder_layers.append(output_layer)
    self._state_means, self._state_mags = get_spec_means_mags(
        self._state_spec)
    self._latent_dim = latent_dim

  @property
  def state_spec(self):
    return self._state_spec

  def forward(self, state):
    h = state
    for l in self._encoder_layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    std = tf.exp(tf.clip_by_value(log_std, -4, 15))
    z = mean + std * tf.random.normal(shape=std.shape)
    s = self.decode(z)
    return s, mean, std

  def decode(self, z):
    h = z
    for l in self._decoder_layers:
      h = l(h)
    if self._state_mags.numpy().mean() == np.inf:
      return h
    else:
      return tf.tanh(h) * self._state_mags + self._state_means

  # def decode_multiple(self, z=None, n=10):
  #   if z is None:
  #     z = tf.random.normal(shape=(n, state.shape[0], self._latent_dim))
  #     z = tf.clip_by_value(z, -0.5, 0.5)
  #   # state_tile = tf.reshape(tf.tile(state, (1, n)), (state.shape[0], n, -1))
  #   state_tile = tf.reshape(tf.tile(state, (n, 1)), (n, state.shape[0], -1))
  #   h = tf.concat([state_tile, z], axis=-1)
  #   for l in self._decoder_layers:
  #     h = l(h)
  #   a = tf.tanh(h) * self._action_mags + self._action_means
  #   # a = tf.clip_by_value(
  #   #     a, self._action_spec.minimum, self._action_spec.maximum)
  #   return a
  #
  # def sample(self, state):
  #   z = tf.random.normal(shape=[state.shape[0], self._latent_dim])
  #   z = tf.clip_by_value(z, -0.5, 0.5)
  #   return self.decode(z)
  #
  # def sample_n(self, state, n=10):
  #   a_n = self.decode_multiple(state, n=n)
  #   return None, a_n, None

  def get_log_density(self, state):
    # variational lower bound
    s_recon, mean, std = self.forward(state)
    # log_2pi = tf.log(tf.constant(math.pi))
    # recon = - 0.5 * tf.reduce_mean(
    #     tf.square(s_recon - state) + log_2pi, axis=-1)
    # kl = 0.5 * tf.reduce_mean(
    #     - 1.0 - tf.log(tf.square(std)) + tf.square(mean) + tf.square(std),
    #     axis=-1)
    recon_loss = tf.reduce_mean(tf.square(s_recon - state), axis=-1)
    kl_losses = -0.5 * (1.0 + tf.log(tf.square(std)) - tf.square(mean) -
                        tf.square(std))
    kl_loss = tf.reduce_mean(kl_losses, axis=-1)
    b_loss = recon_loss + kl_loss * 0.5  # Based on the pytorch implementation.
    return -b_loss

  @property
  def weights(self):
    w_list = []
    for l in self._encoder_layers:
      w_list.append(l.weights[0])
    for l in self._decoder_layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, state):
    return self.forward(state)


class DNNNetwork(tf.Module):
    """Vanilla DNN."""

    def __init__(
            self,
            output_dim,
            fc_layer_params, ):
        super(DNNNetwork, self).__init__()
        self._dnn_layers = []
        for n_units in fc_layer_params:
            l = tf.keras.layers.Dense(
                n_units,
                activation=tf.nn.swish,
            )
            self._dnn_layers.append(l)
        output_layer = tf.keras.layers.Dense(
            output_dim,
            activation=None)
        self._dnn_layers.append(output_layer)

    def forward(self, state):
        h = state
        for l in self._dnn_layers:
            h = l(h)
        return h

    @property
    def weights(self):
        w_list = []
        for l in self._dnn_layers:
            w_list.append(l.weights[0])
        return w_list

    def __call__(self, state):
        return self.forward(state)


class ADMBehavior(tf.Module):
    """ADM network: for behavior policy"""
    def __init__(self,
                 action_spec,
                 fc_layer_params=(),
                 out_reranking=None,
                 latent_dim=None):
        super().__init__()
        self._action_spec = action_spec
        if latent_dim is None:
            latent_dim = action_spec.shape[0] * 2
        self.out_dim = action_spec.shape[0]
        if out_reranking is None:
            self.out_rank = np.arange(self.out_dim)
        else:
            self.out_rank = out_reranking
        self.shared_layer_params = None
        self.indiv_layer_params = None
        if isinstance(fc_layer_params[0], (list, tuple)):
            assert len(fc_layer_params) == 2
            self.shared_layer_params = fc_layer_params[0]
            self.indiv_layer_params = fc_layer_params[-1]
            self.shared_layer = []
            for n_units in self.shared_layer_params:
                l = tf.keras.layers.Dense(
                    n_units,
                    activation=tf.nn.relu,
                )
                self.shared_layer.append(l)
            output_layer = tf.keras.layers.Dense(
                latent_dim,
                activation=None,
                name='latent_out')
            self.shared_layer.append(output_layer)
        elif isinstance(fc_layer_params[0], int):
            self.indiv_layer_params = fc_layer_params
        self._layers_list = []
        for i in range(self.out_dim):
            _layers = []
            for n_units in self.indiv_layer_params:
                l = tf.keras.layers.Dense(
                    n_units,
                    activation=tf.nn.relu
                )
                _layers.append(l)
            output_layer = tf.keras.layers.Dense(
                2,
                activation=None,
                name=f'out_{i}')  # mean and std of single dimension
            _layers.append(output_layer)
            self._layers_list.append(_layers)
        self._action_means, self._action_mags = get_spec_means_mags(action_spec)

    @property
    def action_spec(self):
        return self._action_spec

    def _get_outputs(self, state, _layers):
        h = state
        for l in _layers:
            h = l(h)
        mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
        a_tanh_mode = tf.tanh(mean) * self._action_mags + self._action_means
        log_std = tf.tanh(log_std)
        log_std = (log_std + 1) * 0.5 * (LOG_STD_MAX - LOG_STD_MIN) + LOG_STD_MIN
        std = tf.exp(log_std)
        a_distribution = tfd.TransformedDistribution(
            distribution=tfd.Normal(loc=0.0, scale=1.0),
            bijector=tfp.bijectors.Chain([
                tfp.bijectors.AffineScalar(shift=self._action_means, scale=self._action_mags),
                tfp.bijectors.Tanh(),
                tfp.bijectors.AffineScalar(shift=mean, scale=std),
            ]),
            event_shape=[tf.shape(mean)[-1]],
            batch_shape=[tf.shape(mean)[0]]
        )
        return a_distribution, a_tanh_mode

    def __call__(self, s):
        if self.shared_layer_params is None:
            h = s
        else:
            h = s
            for l in self.shared_layer:
                h = l(h)
        batch_size = s.shape[0]
        out_tanh_mode = tf.zeros(shape=[batch_size, self.out_dim])
        outs_sample = tf.zeros(shape=[batch_size, self.out_dim])
        log_pi_outs = tf.zeros(shape=[batch_size, self.out_dim])
        outs_dist = [None] * self.out_dim
        out_middle = None
        for i, index in enumerate(self.out_rank):
            if i == 0:
                out_dist, out = self._get_outputs(h, self._layers_list[i])
                out_middle = tf.concat([h, out], axis=-1)
            else:
                out_dist, out = self._get_outputs(out_middle, self._layers_list[i])
                out_middle = tf.concat([out_middle, out], axis=-1)
            out_sample = out_dist.sample()
            log_pi_out = out_dist.log_prob(out_sample)

            out_tanh_mode = tensor_modified(out_tanh_mode,
                                            out,
                                            [index, index + 1])
            outs_sample = tensor_modified(outs_sample,
                                          out_sample,
                                          [index, index + 1])
            log_pi_outs = tensor_modified(log_pi_outs,
                                          tf.reshape(log_pi_out, (-1, 1)),
                                          [index, index + 1])
            outs_dist[index] = out_dist

        return out_tanh_mode, outs_sample, log_pi_outs, outs_dist

    def get_log_density(self, state, action):
        _, _, log_pi_out_pred, outs_dist = self(state)
        log_density = []
        for i, out_dist in enumerate(outs_dist):
            log_density.append(out_dist.log_prob(action[:, i:i+1]))

        return tf.transpose(log_density), log_pi_out_pred

    @property
    def weights(self):
        w_list = []
        if self.shared_layer_params is not None:
            for l in self.shared_layer:
                w_list.append(l.weights[0])
        for _layers in self._layers_list:
            for l in _layers:
                w_list.append(l.weights[0])

        return w_list

    def sample_n(self, state, n=1):
        repeat_state = tf.tile(state, (n, 1))
        _, outs_sample, _, _ = self(repeat_state)

        return outs_sample, tf.reshape(outs_sample, (n, state.shape[0], -1))

    def sample(self, state):
        return self.sample_n(state, n=1)[0]


class ADMDynamic(tf.Module):
    """ADM network: for Dynamic model(s1+a1 --> s2+r)"""

    def __init__(self,
                 state_dim,
                 fc_layer_params=(),
                 out_reranking=None,
                 latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = state_dim
        self.out_dim = state_dim + 1  # state + reward
        if out_reranking is None:
            self.out_rank = np.arange(self.out_dim)
        else:
            self.out_rank = out_reranking
        self.shared_layer_params = None
        self.indiv_layer_params = None
        if isinstance(fc_layer_params[0], (list, tuple)):
            assert len(fc_layer_params) == 2
            self.shared_layer_params = fc_layer_params[0]
            self.indiv_layer_params = fc_layer_params[-1]
            self.shared_layer = []
            for n_units in self.shared_layer_params:
                l = tf.keras.layers.Dense(
                    n_units,
                    activation=tf.nn.relu,
                )
                self.shared_layer.append(l)
            output_layer = tf.keras.layers.Dense(
                latent_dim,
                activation=None,
                name='latent_out')
            self.shared_layer.append(output_layer)
        elif isinstance(fc_layer_params[0], int):
            self.indiv_layer_params = fc_layer_params
        self._layers_list = []
        for i in range(self.out_dim):
            _layers = []
            for n_units in self.indiv_layer_params:
                l = tf.keras.layers.Dense(
                    n_units,
                    activation=tf.nn.relu
                )
                _layers.append(l)
            output_layer = tf.keras.layers.Dense(
                2,
                activation=None,
                name=f'out_{i}')  # mean and std of single dimension
            _layers.append(output_layer)
            self._layers_list.append(_layers)

    def _get_outputs(self, inputs, _layers):
        h = inputs
        for l in _layers:
            h = l(h)
        mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
        log_std = tf.tanh(log_std)
        log_std = (log_std + 1) * 0.5 * (LOG_STD_MAX - LOG_STD_MIN) + LOG_STD_MIN
        std = tf.exp(log_std)
        a_distribution = tfd.TransformedDistribution(
            distribution=tfd.Normal(loc=0.0, scale=1.0),
            bijector=tfp.bijectors.Chain([
                tfp.bijectors.AffineScalar(shift=mean, scale=std),
            ]),
            event_shape=[tf.shape(mean)[-1]],
            batch_shape=[tf.shape(mean)[0]]
        )
        return a_distribution, mean

    def __call__(self, s, a):
        if self.shared_layer_params is None:
            h = tf.concat([s, a], axis=-1)
        else:
            h = tf.concat([s, a], axis=-1)
            for l in self.shared_layer:
                h = l(h)
        batch_size = s.shape[0]
        out_mean = tf.zeros(shape=[batch_size, self.out_dim])
        outs_sample = tf.zeros(shape=[batch_size, self.out_dim])
        log_pi_outs = tf.zeros(shape=[batch_size, self.out_dim])
        outs_dist = [None] * self.out_dim
        out_middle = None
        for i, index in enumerate(self.out_rank):
            if i == 0:
                out_dist, out = self._get_outputs(h, self._layers_list[i])
                out_middle = tf.concat([h, out], axis=-1)
            else:
                out_dist, out = self._get_outputs(out_middle, self._layers_list[i])
                out_middle = tf.concat([out_middle, out], axis=-1)
            out_sample = out_dist.sample()
            log_pi_out = out_dist.log_prob(out_sample)

            out_mean = tensor_modified(out_mean,
                                       out,
                                       [index, index + 1])
            outs_sample = tensor_modified(outs_sample,
                                          out_sample,
                                          [index, index + 1])
            log_pi_outs = tensor_modified(log_pi_outs,
                                          tf.reshape(log_pi_out, (-1, 1)),
                                          [index, index + 1])
            outs_dist[index] = out_dist

        return out_mean, outs_sample, log_pi_outs, outs_dist

    def get_log_density(self, state, action, next_state, reward):
        _target = tf.concat([next_state, tf.reshape(reward, (-1, 1))], axis=-1)
        _, _, log_pi_out_pred, outs_dist = self(state, action)
        log_density = []
        for i, out_dist in enumerate(outs_dist):
            log_density.append(out_dist.log_prob(_target[:, i:i + 1]))

        return tf.transpose(log_density), log_pi_out_pred

    @property
    def weights(self):
        w_list = []
        if self.shared_layer_params is not None:
            for l in self.shared_layer:
                w_list.append(l.weights[0])
        for _layers in self._layers_list:
            for l in _layers:
                w_list.append(l.weights[0])

        return w_list

    def sample_n(self, state, action, n=1):
        repeat_state = tf.tile(state, (n, 1))
        repeat_action = tf.tile(action, (n, 1))
        _, outs_sample, _, _ = self(repeat_state, repeat_action)

        return outs_sample, tf.reshape(outs_sample, (n, state.shape[0], -1))

    def sample(self, state, action):
        return self.sample_n(state, action, n=1)[0]


def tensor_modified(a, b, index):
    return tf.concat([a[:, :index[0]], b, a[:, index[1]:]], axis=-1)


class ProbDynamics(tf.Module):
  """Probabilistic dynamics model.

  References:
      * `Janner et al., When to Trust Your Model: Model-Based Policy \
        Optimization. <https://arxiv.org/abs/1906.08253>`_
      * `Chua et al., Deep Reinforcement Learning in a Handful of Trials \
        using Probabilistic Dynamics Models. \
        <https://arxiv.org/abs/1805.12114>`_

  :param int state_dim: the dimension of the state.
  :param fc_layer_params: the parameter of the network structure.
  :param str mode: 'local' means that this model predicts the difference to the
      current state.
  """

  def __init__(self, state_dim, fc_layer_params=(), mode='local'):
    super(ProbDynamics, self).__init__()
    self._state_dime = state_dim
    output_dim = state_dim + 1  # add the dimension of the reward.
    self._layers = []
    for n_units in fc_layer_params:
      l = tf.keras.layers.Dense(
        n_units,
        activation=tf.nn.swish,
      )
      self._layers.append(l)
    output_layer = tf.keras.layers.Dense(
      output_dim * 2,
      activation=None,
    )
    self._layers.append(output_layer)
    self._max_logstd = tf.Variable(np.ones([1, output_dim]) * 1, dtype=tf.float32, name='max_log_std', trainable=True)
    self._min_logstd = tf.Variable(np.ones([1, output_dim]) * (-5), dtype=tf.float32, name='min_log_std', trainable=True)
    self._mode = mode

  def _get_outputs(self, s, a):
    """The last dimension of the output is reward."""
    h = tf.concat([s, a], axis=-1)
    for l in self._layers:
      h = l(h)
    mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
    log_std = self._max_logstd - tf.nn.softplus(self._max_logstd - log_std)
    log_std = self._min_logstd + tf.nn.softplus(log_std - self._min_logstd)
    std = tf.exp(log_std)
    if self._mode == 'local':
      obs, reward = tf.split(mean, [self._state_dime, 1], axis=-1)
      obs = obs + s
      mean = tf.concat([obs, reward], axis=-1)
    dist = tfd.Normal(loc=mean, scale=std)
    return dist

  def get_log_density(self, s1, a1, s2, r):
    """Compute the log probability density of the real target
    in the predict state distribution.

    :param s1: the current state.
    :param a1: the current action.
    :param s2: the next state.
    :param r: the reward.
    :return log_density: log probability density.
    """
    dist = self._get_outputs(s1, a1)
    r = tf.reshape(r, (-1, 1))
    log_density = dist.log_prob(tf.concat([s2, r], axis=-1))
    return log_density

  @property
  def weights(self):
    w_list = []
    for l in self._layers:
      w_list.append(l.weights[0])
    return w_list

  def __call__(self, s, a):
    dist = self._get_outputs(s, a)
    mean = dist.mean()
    sample = dist.sample()
    return mean, sample, dist

  @property
  def max_logstd(self):
    return self._max_logstd

  @property
  def min_logstd(self):
    return self._min_logstd

