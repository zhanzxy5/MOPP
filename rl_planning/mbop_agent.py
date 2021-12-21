"""
Implementation of MBOP.

Based on 'Model-based offline planning' by Arthur Argenson and Gabriel Dulac-Arnold.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf
import numpy as np
import agent
from scripts import networks
from scripts import utils

CLIP_EPS = 1e-3


@gin.configurable
class Agent(agent.Agent):
    """MBOP agent."""

    def __init__(self,
                 use_value_fn=True,
                 b_list=None,
                 d_list=None,
                 pred_len=10,
                 beta=0.6,
                 pop_size=1000,
                 kappa=0.9,
                 noise_sigma=0.08,
                 model_id=0,
                 test_b_only=False,
                 **kwargs):
        self._use_value_fn = use_value_fn  # if add the predicted return to the cumulative rewards
        self._b_list = b_list
        self._d_list = d_list
        self.model_id = model_id
        self._test_b_only = test_b_only
        # offline MPPI parameters
        self.pred_len = pred_len
        self.beta = beta
        self.pop_size = pop_size
        self.kappa = kappa
        self.noise_sigma = noise_sigma
        self.prev_sol = None
        super(Agent, self).__init__(**kwargs)
        self.past_action = np.zeros(self._a_dim)
        self._a_min = float(self._action_spec.minimum)

    def _build_fns(self):
        self._agent_module = AgentModule(modules=self._modules)
        self._b_fns = self._agent_module.b_nets
        self._d_fns = self._agent_module.d_nets
        self._q_fns = self._agent_module.q_nets
        if self._b_list is None:
            self._b_list = np.arange(len(self._b_fns))
        if self._d_list is None:
            self._d_list = np.arange(len(self._d_fns))
        self.dynamic = self._get_dynamic([self._agent_module.d_nets[i] for i in self._d_list])
        self.behavior = self._get_behavior([self._agent_module.b_nets[i] for i in self._b_list])
        self.value_fn = self._get_value_fn(self._agent_module.q_nets)

    def _get_b_vars(self):
        return self._agent_module.b_variables

    def _get_d_vars(self):
        return self._agent_module.d_variables

    def _get_q_vars(self):
        return self._agent_module.q_variables

    def _get_b_weight_norm(self):
        weights = self._agent_module.b_weights
        norms = []
        for w in weights:
            norm = tf.reduce_sum(tf.square(w))
            norms.append(norm)
        return tf.add_n(norms)

    def _get_d_weight_norm(self):
        weights = self._agent_module.d_weights
        norms = []
        for w in weights:
            norm = tf.reduce_sum(tf.square(w))
            norms.append(norm)
        return tf.add_n(norms)

    def _get_q_weight_norm(self):
        weights = self._agent_module.q_weights
        norms = []
        for w in weights:
            norm = tf.reduce_sum(tf.square(w))
            norms.append(norm)
        return tf.add_n(norms)

    def _build_b_loss(self, batch):
        a0 = batch['a1']
        s = batch['s2']
        a_b = batch['a2']
        a_b = utils.clip_by_eps(a_b, self._action_spec, CLIP_EPS)
        b_losses = []
        for b_fn in self._b_fns:
            input_ = tf.concat((s, a0), axis=-1)
            a_p = b_fn(input_)
            b_loss_ = tf.reduce_mean(tf.square(a_p - a_b))
            b_losses.append(b_loss_)
        b_loss = tf.add_n(b_losses)
        b_w_norm = self._get_b_weight_norm()
        norm_loss = self._weight_decays * b_w_norm
        loss = b_loss + norm_loss

        info = collections.OrderedDict()
        info['b_loss'] = b_loss
        info['b_norm'] = b_w_norm
        return loss, info

    def _build_d_loss(self, batch):
        s1 = batch['s1']
        s2 = batch['s2']
        a1 = batch['a1']
        r = batch['r']
        d_losses = []
        for d_fn in self._d_fns:
            input_ = tf.concat((s1, a1), axis=-1)
            output_ = d_fn(input_)
            target_ = tf.concat([s2, tf.reshape(r, (-1, 1))], axis=-1)
            d_loss_ = tf.reduce_mean(tf.square(output_ - target_))
            d_losses.append(d_loss_)
        d_loss = tf.add_n(d_losses)
        d_w_norm = self._get_d_weight_norm()
        norm_loss = self._weight_decays * d_w_norm
        loss = d_loss + norm_loss

        info = collections.OrderedDict()
        info['d_loss'] = d_loss
        info['d_norm'] = d_w_norm
        return loss, info

    # to be update
    def _build_q_loss(self, batch_list):
        assert len(batch_list) == self.pred_len + 1
        batch = batch_list[0]
        s2 = batch['s2']
        a1 = batch['a1']
        # Compute the truncated value function target.
        q_target = []
        for batch_ in batch_list[1:]:
            q_target.append(batch_['r'])
        q_target = tf.reduce_sum(q_target, axis=0)
        q_losses = []
        for q_fn in self._q_fns:
            q_pred = q_fn(s2, a1)
            q_loss_ = tf.reduce_mean(tf.square(q_pred - q_target))
            q_losses.append(q_loss_)
        q_loss = tf.add_n(q_losses)
        q_w_norm = self._get_q_weight_norm()
        norm_loss = self._weight_decays * q_w_norm
        loss = q_loss + norm_loss

        info = collections.OrderedDict()
        info['q_loss'] = q_loss
        info['q_norm'] = q_w_norm
        return loss, info

    def _build_optimizers(self):
        opts = self._optimizers
        self._b_optimizer = utils.get_optimizer(opts[0][0])(lr=opts[0][1])
        self._d_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])
        self._q_optimizer = utils.get_optimizer(opts[-1][0])(lr=opts[-1][1])

    @tf.function
    def _optimize_step(self, batch):
        pass

    @tf.function
    def _optimize_b(self, batch):
        vars_ = self._b_vars
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(vars_)
            loss, info = self._build_b_loss(batch)
        grads = tape.gradient(loss, vars_)
        grads_and_vars = tuple(zip(grads, vars_))
        self._b_optimizer.apply_gradients(grads_and_vars)
        return info

    @tf.function
    def _optimize_d(self, batch):
        vars_ = self._d_vars
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(vars_)
            loss, info = self._build_d_loss(batch)
        grads = tape.gradient(loss, vars_)
        grads_and_vars = tuple(zip(grads, vars_))
        self._d_optimizer.apply_gradients(grads_and_vars)
        return info

    @tf.function
    def _optimize_q(self, batch_list):
        vars_ = self._q_vars
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(vars_)
            loss, info = self._build_q_loss(batch_list)
        grads = tape.gradient(loss, vars_)
        grads_and_vars = tuple(zip(grads, vars_))
        self._q_optimizer.apply_gradients(grads_and_vars)
        return info

    def _init_vars(self, batch):
        self._build_b_loss(batch)
        self._build_d_loss(batch)
        batch_q = self._get_train_batch_q()
        self._build_q_loss(batch_q)
        self._b_vars = self._get_b_vars()
        self._d_vars = self._get_d_vars()
        self._q_vars = self._get_q_vars()

    def _build_checkpointer(self):
        state_ckpt = tf.train.Checkpoint(
            agent=self._agent_module,
            global_step=self._global_step,
        )
        behavior_ckpt = tf.train.Checkpoint(
            policy=self._agent_module.b_nets,
        )
        dynamics_ckpt = tf.train.Checkpoint(
            dynamics=self._agent_module.d_nets,
        )
        q_ckpt = tf.train.Checkpoint(
            q=self._agent_module.q_nets,
        )
        return dict(
            state=state_ckpt,
            behavior=behavior_ckpt,
            dynamics=dynamics_ckpt,
            q_fn=q_ckpt)

    def train_behavior_step(self):
        train_b_info = collections.OrderedDict()
        train_batch = self._get_train_batch()
        info = self._optimize_b(train_batch)
        for key, val in info.items():
            train_b_info[key] = val.numpy()
        return train_b_info

    @tf.function
    def test_behavior(self, _batch):
        a0 = _batch['a1']
        s = _batch['s2']
        a_b = _batch['a2']
        test_b_info = collections.OrderedDict()
        for i, b_fn in enumerate(self._b_fns):
            a_p = b_fn(tf.concat((s, a0), axis=-1))
            a_error = tf.reduce_mean(tf.square(a_p - a_b))
            test_b_info[f'b{i}_mse'] = a_error
        return test_b_info

    def write_b_train_summary(self, summary_writer, step, info):
        _batch = self._get_train_batch()
        test_b_info = self.test_behavior(_batch)
        for key, val in test_b_info.items():
            info[key] = val.numpy()
        utils.write_summary(summary_writer, step, info)

    def save_behavior_model(self, ckpt_dir):
        self._checkpointer['behavior'].write(ckpt_dir)

    def restore_behavior_model(self, ckpt_dir):
        self._checkpointer['behavior'].restore(ckpt_dir)

    def train_dynamics_step(self):
        train_d_info = collections.OrderedDict()
        train_batch = self._get_train_batch()
        info = self._optimize_d(train_batch)
        for key, val in info.items():
            train_d_info[key] = val.numpy()
        return train_d_info

    @tf.function
    def test_dynamics(self, _batch):
        s1 = _batch['s1']
        s2 = _batch['s2']
        a1 = _batch['a1']
        r = _batch['r']
        test_d_info = collections.OrderedDict()
        for i, d_fn in enumerate(self._d_fns):
            _p = d_fn(tf.concat((s1, a1), axis=-1))
            s_p = _p[:, :-1]
            r_p = _p[:, -1]
            s_error = tf.reduce_mean(tf.square(s_p - s2))
            r_error = tf.reduce_mean(tf.square(r_p - r))
            test_d_info[f'd{i}_s_mse'] = s_error
            test_d_info[f'd{i}_r_mse'] = r_error
        return test_d_info

    def write_d_train_summary(self, summary_writer, step, info):
        _batch = self._get_train_batch()
        test_d_info = self.test_dynamics(_batch)
        for key, val in test_d_info.items():
            info[key] = val.numpy()
        utils.write_summary(summary_writer, step, info)

    def save_dynamics_model(self, ckpt_dir):
        self._checkpointer['dynamics'].write(ckpt_dir)

    def restore_dynamics_model(self, ckpt_dir):
        self._checkpointer['dynamics'].restore(ckpt_dir)

    def train_q_step(self, step):
        train_q_info = collections.OrderedDict()
        train_batch_list = self._get_train_batch_q()
        info = self._optimize_q(train_batch_list)
        for key, val in info.items():
            train_q_info[key] = val.numpy()
        return train_q_info

    def _get_train_batch_q(self):
        """Get the train batch for a truncated value function."""
        batch_indices = np.random.choice(self._train_data.size-self.pred_len, self._batch_size)
        train_batch_q = [self._get_batch(batch_indices)]
        for i in range(self.pred_len):
            batch_indices += 1
            train_batch_q.append(self._get_batch(batch_indices))
        return train_batch_q

    def save_q_model(self, ckpt_dir):
        self._checkpointer['q_fn'].write(ckpt_dir)

    def restore_q_model(self, ckpt_dir):
        self._checkpointer['q_fn'].restore(ckpt_dir)

    def test_behavior_all_data(self):
        batch_num = int(self._train_data.size / self._batch_size)
        a_pred, a_real = [], []
        for i in range(batch_num):
            _batch = self._get_batch(np.arange(i * self._batch_size, i * self._batch_size + self._batch_size))
            a1 = _batch['a1']
            s2 = _batch['s2']
            a2 = _batch['a2']
            a_pred.append(self.behavior(s2, a1, id=self.model_id).numpy())
            a_real.append(a2.numpy())
        a_mse = np.mean(np.square(np.array(a_pred) - np.array(a_real)))
        return a_mse

    def test_dynamic_all_data(self):
        batch_num = int(self._train_data.size / self._batch_size)
        s_pred, r_pred, s_real, r_real = [], [], [], []
        for i in range(batch_num):
            _batch = self._get_batch(np.arange(i * self._batch_size, i * self._batch_size + self._batch_size))
            s1 = _batch['s1']
            s2 = _batch['s2']
            a1 = _batch['a1']
            r = _batch['r']
            s_p, r_p = self.dynamic(s1, a1, id=self.model_id)
            s_pred.append(s_p.numpy())
            r_pred.append(r_p.numpy())
            s_real.append(s2.numpy())
            r_real.append(r.numpy())
        s_mse = np.mean(np.square(np.array(s_pred) - np.array(s_real)))
        r_mse = np.mean(np.square(np.array(r_pred) - np.array(r_real)))
        return s_mse, r_mse

    def _build_test_policies(self):
        if self._test_b_only:
            self._test_policies['bc'] = self._behavior_policy
        else:
            self._test_policies['mbop'] = self._mbop_sol

    def _behavior_policy(self, observation, action0,  state=()):
        return self.behavior(observation, action0, np.random.randint(0, len(self._b_list))), state

    def _get_value_fn(self, q_fn):
        @tf.function
        def _value_fn(s, a0):
            out = []
            for q_ in q_fn:
                out.append(q_(s, a0))
            return tf.reduce_mean(out, axis=0)
        return _value_fn

    def _get_behavior(self, behavior):
        @tf.function
        def _behavior(s, a0, id=0):
            input_ = tf.concat((s, a0), axis=-1)
            return behavior[id](input_)

        return _behavior

    def _get_dynamic(self, dynamic):
        @tf.function
        def _dynamic(s, a, id=0):
            s1 = []
            input_ = tf.concat((s, a), axis=-1)
            for d_ in dynamic:
                s1.append(d_(input_))
            return s1[id][:, :-1], tf.reduce_mean(s1, axis=0)[:, -1]

        return _dynamic

    def _clear_sol(self):
        """ clear prev sol"""
        self.prev_sol = None

    def _init_sol(self):
        """Initialize the actions using the behavior policy"""
        init_sol = np.zeros(shape=(self.pred_len, self._a_dim))
        return init_sol

    # MBOP
    def _mbop_sol(self, observation, state=()):
        """ calculate the optimal inputs"""
        if len(observation.shape) == 2:
            assert observation.shape[0] == 1
            curr_x = observation[0]
        else:
            assert len(observation.shape) == 1
            curr_x = observation
        # get prev_sol
        if self.prev_sol is None:
            self.prev_sol = self._init_sol()
        # get noised inputs
        noise = np.random.normal(
            loc=0, scale=1.0, size=(self.pop_size, self.pred_len,
                                    self._a_dim)) * self.noise_sigma
        noise = noise.astype('float32')
        noised_inputs = noise.copy()
        s = np.tile(curr_x, (self.pop_size, 1))
        s_list = np.array_split(s, len(self._b_list))
        s_list = [tf.convert_to_tensor(s, dtype=tf.float32) for s in s_list]
        a0 = self.prev_sol[0]
        a0 = np.tile(a0, (self.pop_size, 1))
        a0_list = np.array_split(a0, len(self._b_list))
        a0_list = [tf.convert_to_tensor(a0, dtype=tf.float32) for a0 in a0_list]
        noise_list = np.array_split(noise, len(self._b_list))
        noised_inputs_list = np.array_split(noised_inputs, len(self._b_list))
        rewards = []
        s_last = []
        # Use consistent ensemble head throughout trajectory.
        for l, s in enumerate(s_list):
            a0, noise, noised_inputs = a0_list[l], noise_list[l], noised_inputs_list[l]
            reward = 0
            for t in range(self.pred_len):
                a0 = self.behavior(s, a0, id=l) + noise[:, t, :]
                a_sample_mix = self.beta * self.prev_sol[t, :] + (1 - self.beta) * a0.numpy()
                noised_inputs[:, t, :] = a_sample_mix
                s, r = self.dynamic(s, tf.convert_to_tensor(a_sample_mix, dtype=tf.float32), id=l)
                reward += r.numpy()
            rewards.append(reward)
            s_last.append(s)
        # clip actions
        noised_inputs = np.concatenate(noised_inputs_list, axis=0)
        noised_inputs = np.clip(noised_inputs, self._a_min, self._a_max)
        rewards = np.concatenate(rewards)
        # calc reward
        if self._use_value_fn:
            s_last = tf.concat(s_last, axis=0)
            a_last = tf.convert_to_tensor(noised_inputs[:, -1, :], dtype=tf.float32)
            q_last = self.value_fn(s_last, a_last).numpy()
            rewards += q_last
        # mppi update
        # normalize and get sum of reward
        # exp_rewards.shape = (N, )
        exp_rewards = np.exp(self.kappa * (rewards - np.max(rewards)))
        denom = np.sum(exp_rewards) + 1e-10  # avoid numeric error
        # weight actions
        weighted_inputs = exp_rewards[:, np.newaxis, np.newaxis] \
                          * noised_inputs
        sol = np.sum(weighted_inputs, 0) / denom
        # update
        self.prev_sol[:-1] = sol[1:]
        self.prev_sol[-1] = sol[-1]  # last use the terminal input
        # self.past_action = sol[0]

        return sol[0][np.newaxis, :], state


class AgentModule(agent.AgentModule):
    """Tensorflow module for MBOP agent."""

    def _build_modules(self):
        self._b_nets = []
        self._d_nets = []
        self._q_nets = []
        n_b_fns = self._modules.n_b_fns
        n_d_fns = self._modules.n_d_fns
        n_q_fns = self._modules.n_q_fns
        for _ in range(n_b_fns):
            self._b_nets.append(
                self._modules.b_net_factory()
            )
        for _ in range(n_d_fns):
            self._d_nets.append(
                self._modules.d_net_factory()
            )
        for _ in range(n_q_fns):
            self._q_nets.append(
                self._modules.q_net_factory()
            )

    @property
    def b_nets(self):
        return self._b_nets

    @property
    def b_weights(self):
        b_weights = []
        for b_net in self._b_nets:
            b_weights += b_net.weights
        return b_weights

    @property
    def b_variables(self):
        vars_ = []
        for b_net in self._b_nets:
            vars_ += b_net.trainable_variables
        return tuple(vars_)

    @property
    def d_nets(self):
        return self._d_nets

    @property
    def d_weights(self):
        d_weights = []
        for d_net in self._d_nets:
            d_weights += d_net.weights
        return d_weights

    @property
    def d_variables(self):
        vars_ = []
        for d_net in self._d_nets:
            vars_ += d_net.trainable_variables
        return tuple(vars_)

    @property
    def q_nets(self):
        return self._q_nets

    @property
    def q_weights(self):
        q_weights = []
        for q_net in self._q_nets:
            q_weights += q_net.weights
        return q_weights

    @property
    def q_variables(self):
        vars_ = []
        for q_net in self._q_nets:
            vars_ += q_net.trainable_variables
        return tuple(vars_)


def get_modules(model_params, action_spec, state_spec):
    """Get agent modules."""
    model_params, _, _ = model_params
    assert len(model_params) == 3

    def b_net_factory():
        return networks.DNNNetwork(
            action_spec.shape[0],
            fc_layer_params=model_params[0][0],
        )

    def d_net_factory():
        return networks.DNNNetwork(
            output_dim=state_spec.shape[0] + 1,
            fc_layer_params=model_params[1][0],
        )

    def q_net_factory():
        return networks.CriticNetwork(
            fc_layer_params=model_params[2][0])

    modules = utils.Flags(
        b_net_factory=b_net_factory,
        d_net_factory=d_net_factory,
        q_net_factory=q_net_factory,
        n_b_fns=model_params[0][1],
        n_d_fns=model_params[1][1],
        n_q_fns=model_params[2][1],
    )
    return modules


class Config(agent.Config):

    def _get_modules(self):
        return get_modules(
            self._agent_flags.model_params,
            self._agent_flags.action_spec,
            self._agent_flags.observation_spec,
        )
