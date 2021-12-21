"""
Implementation of MOPP-BC (Using the BC model as the behavior policy).
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
import tensorflow_probability as tfp

CLIP_EPS = 1e-3


@gin.configurable
class Agent(agent.Agent):

    def __init__(self,
                 train_b_alpha_entropy=False,
                 train_d_alpha_entropy=False,
                 b_alpha_entropy=0.0,
                 d_alpha_entropy=0.0,
                 b_target_entropy=None,
                 d_target_entropy=None,
                 ensemble_q_lambda=1.0,
                 n_action_samples=10,
                 use_value_fn=False,
                 b_list=None,
                 d_list=None,
                 pred_len=10,
                 beta=0.6,
                 pop_size=1000,
                 kappa=0.9,
                 noise_sigma=0.08,
                 behavior_init_coef=0.5,
                 d_uncertainty_threshold=None,
                 uncertainty_percentile=90,
                 penalty_lambda=0,
                 model_id=0,
                 maxq=True,
                 test_b_only=False,
                 **kwargs):
        self._train_b_alpha_entropy = train_b_alpha_entropy
        self._train_d_alpha_entropy = train_d_alpha_entropy
        self._b_alpha_entropy = b_alpha_entropy
        self._d_alpha_entropy = d_alpha_entropy
        self._b_target_entropy = b_target_entropy
        self._d_target_entropy = d_target_entropy
        self._ensemble_q_lambda = ensemble_q_lambda
        self._n_action_samples = n_action_samples
        self._use_value_fn = use_value_fn  # if add the predicted return to the cumulative rewards
        self._b_list = b_list
        self._d_list = d_list
        self.model_id = model_id
        self.maxq = maxq
        self._test_b_only = test_b_only
        # offline MPPI parameters
        self.pred_len = pred_len
        self.beta = beta
        self.pop_size = pop_size
        self.kappa = kappa
        self.noise_sigma = noise_sigma
        self.behavior_init_coef = behavior_init_coef
        self.prev_sol = None
        super(Agent, self).__init__(**kwargs)
        self.past_action = np.zeros(self._a_dim)
        self._a_min = float(self._action_spec.minimum)
        self.d_uncertainty_threshold = d_uncertainty_threshold
        self._uncertainty_percentile = uncertainty_percentile
        self._penalty_lambda = penalty_lambda

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
        if not self.maxq:
            self.behavior_maxq = self._get_behavior_sample([self._agent_module.b_nets[i] for i in self._b_list],
                                                           self.noise_sigma)
        else:
            self.behavior_maxq = self._get_behavior_sample_maxq([self._agent_module.b_nets[i] for i in self._b_list],
                                                                self.noise_sigma)
        self._behavior_sample = self._get_behavior_sample([self._agent_module.b_nets[i] for i in self._b_list], None)
        self.dynamic_uncertainty = self._get_dynamic_diff([self._agent_module.d_nets[i] for i in self._d_list])
        self.value_fn = self._get_value_fn([self._agent_module.q_nets[i][0]
                                            for i in range(len(self._agent_module.q_nets))])
        if self._b_target_entropy is None:
            self._b_target_entropy = - self._action_spec.shape[0]
        if self._d_target_entropy is None:
            self._d_target_entropy = - (self._observation_spec.shape[0] + 1)
        self._get_b_alpha_entropy = self._agent_module.get_b_alpha_entropy
        self._get_d_alpha_entropy = self._agent_module.get_d_alpha_entropy
        self._agent_module.assign_alpha_entropy(self._b_alpha_entropy, self._d_alpha_entropy)

    def _get_b_vars(self):
        return self._agent_module.b_variables

    def _get_d_vars(self):
        return self._agent_module.d_variables

    def _get_q_vars(self):
        return self._agent_module.q_source_variables

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
        weights = self._agent_module.q_source_weights
        norms = []
        for w in weights:
            norm = tf.reduce_sum(tf.square(w))
            norms.append(norm)
        return tf.add_n(norms)

    def ensemble_q(self, qs):
        lambda_ = self._ensemble_q_lambda
        return (lambda_ * tf.reduce_min(qs, axis=-1)
                + (1 - lambda_) * tf.reduce_max(qs, axis=-1))

    def _ensemble_q2_target(self, q2_targets):
        return self.ensemble_q(q2_targets)

    def _build_b_loss(self, batch):
        s = batch['s1']
        a_b = batch['a1']
        a_b = utils.clip_by_eps(a_b, self._action_spec, CLIP_EPS)
        b_losses = []
        for b_fn in self._b_fns:
            log_pi_a_b = b_fn.get_log_density(s, a_b)
            _, _, log_pi_a_p = b_fn(s)
            b_loss_ = tf.reduce_mean(
                self._get_b_alpha_entropy() * log_pi_a_p
                - log_pi_a_b)
            b_losses.append(b_loss_)
        b_loss = tf.add_n(b_losses)
        b_w_norm = self._get_b_weight_norm()
        norm_loss = self._weight_decays * b_w_norm
        loss = b_loss + norm_loss
        # Construct information about current training.
        info = collections.OrderedDict()
        info['b_loss'] = b_loss
        info['b_norm'] = b_w_norm
        return loss, info

    def _build_b_ae_loss(self, batch):
        s = batch['s1']
        b_ae_loss = []
        alpha = self._get_b_alpha_entropy()
        for b_fn in self._b_fns:
            _, _, log_pi_a = b_fn(s)
            b_ae_loss.append(tf.reduce_mean(alpha * (- log_pi_a - self._b_target_entropy)))
        b_ae_loss = tf.reduce_mean(b_ae_loss)
        # Construct information about current training.
        info = collections.OrderedDict()
        info['b_ae_loss'] = b_ae_loss
        info['b_alpha_entropy'] = alpha
        return b_ae_loss, info

    def _build_d_loss(self, batch):
        s1 = batch['s1']
        s2 = batch['s2']
        a1 = batch['a1']
        r = batch['r']
        d_losses = []
        for d_fn in self._d_fns:
            log_pi_s_real, log_pi_s_pred = d_fn.get_log_density(s1, a1, s2, r)
            log_pi_s_real = tf.reduce_sum(log_pi_s_real, axis=-1)
            log_pi_s_pred = tf.reduce_sum(log_pi_s_pred, axis=-1)
            d_loss_ = tf.reduce_mean(self._get_d_alpha_entropy() * log_pi_s_pred - log_pi_s_real)
            d_losses.append(d_loss_)
        d_loss = tf.add_n(d_losses)
        d_w_norm = self._get_d_weight_norm()
        norm_loss = self._weight_decays * d_w_norm
        loss = d_loss + norm_loss

        info = collections.OrderedDict()
        info['d_loss'] = d_loss
        info['d_norm'] = d_w_norm
        return loss, info

    def _build_d_ae_loss(self, batch):
        s = batch['s1']
        a = batch['a1']
        d_ae_loss = []
        alpha = self._get_d_alpha_entropy()
        for d_fn in self._d_fns:
            _, _, log_pi_s_pred, _ = d_fn(s, a)
            log_pi_s_pred = tf.reduce_sum(log_pi_s_pred, axis=-1)
            d_ae_loss.append(tf.reduce_mean(alpha * (- log_pi_s_pred - self._d_target_entropy)))
        d_ae_loss = tf.reduce_mean(d_ae_loss)
        # Construct information about current training.
        info = collections.OrderedDict()
        info['d_ae_loss'] = d_ae_loss
        info['d_alpha_entropy'] = alpha
        return d_ae_loss, info

    def _build_q_loss(self, batch):
        s1 = batch['s1']
        s2 = batch['s2']
        a1 = batch['a1']
        a2_b = batch['a2']
        r = batch['r']
        dsc = batch['dsc']

        q2_targets = []
        q1_preds = []
        for q_fn, q_fn_target in self._q_fns:
            q2_target_ = q_fn_target(s2, a2_b)
            q1_pred = q_fn(s1, a1)
            q1_preds.append(q1_pred)
            q2_targets.append(q2_target_)
        q2_targets = tf.stack(q2_targets, axis=-1)
        q2_target = self._ensemble_q2_target(q2_targets)
        v2_target = q2_target
        q1_target = tf.stop_gradient(r + dsc * self._discount * v2_target)
        q_losses = []
        for q1_pred in q1_preds:
            q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
            q_losses.append(q_loss_)
        q_loss = tf.add_n(q_losses)
        q_w_norm = self._get_q_weight_norm()
        norm_loss = self._weight_decays * q_w_norm
        loss = q_loss + norm_loss

        info = collections.OrderedDict()
        info['q_loss'] = q_loss
        info['q_norm'] = q_w_norm
        info['r_mean'] = tf.reduce_mean(r)
        info['dsc_mean'] = tf.reduce_mean(dsc)
        info['q2_target_mean'] = tf.reduce_mean(q2_target)
        info['q1_target_mean'] = tf.reduce_mean(q1_target)
        return loss, info

    def _get_source_target_vars(self):
        return (self._agent_module.q_source_variables,
                self._agent_module.q_target_variables)

    def _build_optimizers(self):
        opts = self._optimizers
        self._b_optimizer = utils.get_optimizer(opts[0][0])(lr=opts[0][1])
        self._d_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])
        self._b_ae_optimizer = utils.get_optimizer(opts[2][0])(lr=opts[2][1])
        self._d_ae_optimizer = utils.get_optimizer(opts[3][0])(lr=opts[3][1])
        self._q_optimizer = utils.get_optimizer(opts[4][0])(lr=opts[4][1])

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
    def _optimize_q(self, batch, step):
        if tf.equal(step % self._update_freq, 0):
            source_vars, target_vars = self._get_source_target_vars()
            self._update_target_fns(source_vars, target_vars)
        vars_ = self._q_vars
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(vars_)
            loss, info = self._build_q_loss(batch)
        grads = tape.gradient(loss, vars_)
        grads_and_vars = tuple(zip(grads, vars_))
        self._q_optimizer.apply_gradients(grads_and_vars)
        return info

    @tf.function
    def _optimize_b_ae(self, batch):
        vars_ = self._b_ae_vars
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(vars_)
            loss, info = self._build_b_ae_loss(batch)
        grads = tape.gradient(loss, vars_)
        grads_and_vars = tuple(zip(grads, vars_))
        self._b_ae_optimizer.apply_gradients(grads_and_vars)
        return info

    @tf.function
    def _optimize_d_ae(self, batch):
        vars_ = self._d_ae_vars
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(vars_)
            loss, info = self._build_d_ae_loss(batch)
        grads = tape.gradient(loss, vars_)
        grads_and_vars = tuple(zip(grads, vars_))
        self._d_ae_optimizer.apply_gradients(grads_and_vars)
        return info

    def _init_vars(self, batch):
        self._build_b_loss(batch)
        self._build_d_loss(batch)
        # self._build_d_loss_dnn(batch)
        self._build_q_loss(batch)
        self._b_vars = self._get_b_vars()
        self._d_vars = self._get_d_vars()
        self._q_vars = self._get_q_vars()
        self._b_ae_vars = self._agent_module.b_ae_variables
        self._d_ae_vars = self._agent_module.d_ae_variables

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
        if self._train_b_alpha_entropy:
            ae_info = self._optimize_b_ae(train_batch)
            info.update(ae_info)
        for key, val in info.items():
            train_b_info[key] = val.numpy()
        return train_b_info

    @tf.function
    def test_behavior(self, _batch):
        s = _batch['s1']
        a_b = _batch['a1']
        test_b_info = collections.OrderedDict()
        for i, b_fn in enumerate(self._b_fns):
            a_p, _, _ = b_fn(s)
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
        if self._train_d_alpha_entropy:
            ae_info = self._optimize_d_ae(train_batch)
            info.update(ae_info)
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
            _p = d_fn(s1, a1)[0]
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
        step = tf.constant(step, dtype=tf.int64)
        train_q_info = collections.OrderedDict()
        train_batch = self._get_train_batch()
        info = self._optimize_q(train_batch, step)
        for key, val in info.items():
            train_q_info[key] = val.numpy()
        return train_q_info

    def save_q_model(self, ckpt_dir):
        self._checkpointer['q_fn'].write(ckpt_dir)

    def restore_q_model(self, ckpt_dir):
        self._checkpointer['q_fn'].restore(ckpt_dir)

    def test_behavior_all_data(self):
        batch_num = int(self._train_data.size / self._batch_size)
        a_pred, a_real = [], []
        for i in range(batch_num):
            _batch = self._get_batch(np.arange(i * self._batch_size, i * self._batch_size + self._batch_size))
            s1 = _batch['s1']
            a1 = _batch['a1']
            a_pred.append(self.behavior(s1, id=self.model_id).numpy())
            a_real.append(a1.numpy())
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
            s_p, r_p = self.dynamic(s1, a1, id=0)
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
            self._test_policies['MOPP-BC'] = self._mopp_bc_sol

    def _behavior_policy(self, observation, state=()):
        return self.behavior(observation, np.random.randint(0, len(self._b_list))), state

    def _get_value_fn(self, q_fn):
        @tf.function
        def _value_fn(s, a):
            out = []
            for q_ in q_fn:
                out.append(q_(s, a))
            return tf.reduce_mean(out, axis=0)
        return _value_fn

    def _get_behavior(self, behavior):
        @tf.function
        def _behavior(s, id=0):
            return behavior[id](s)[0]

        return _behavior

    def _get_behavior_maxq(self, behavior, sigma):
        sigma = tf.constant(sigma, dtype=tf.float32)
        q_fn = self._q_fns[0][0]
        @tf.function
        def _behavior(s, id=0):
            b = behavior
            a = b(s)[0]
            s_n = tf.tile(s, [self._n_action_samples, 1])
            a_n = tf.tile(a, [self._n_action_samples, 1])
            a_n += tf.random.normal(shape=a_n.shape, mean=0.0, stddev=1.0) * sigma
            q_n = q_fn(s_n, a_n)
            q_n = tf.reshape(q_n, (self._n_action_samples, s.shape[0]))  # shape: (n_sample, batch_size)
            a_indices = tf.argmax(q_n, axis=0)
            gather_indices = tf.stack(
                [a_indices, tf.range(s.shape[0], dtype=tf.int64)], axis=-1
            )
            actions = tf.reshape(a_n, (self._n_action_samples, s.shape[0], -1))  # shape: (n_sample, batch_size, a_dim)
            action = tf.gather_nd(actions, gather_indices)
            return action
        return _behavior

    def _get_behavior_sample_maxq(self, behavior, sigma):
        """Scaling the stddev of the behavior"""
        tfd = tfp.distributions
        sigma = tf.constant(sigma, dtype=tf.float32)
        q_fn = self._q_fns[0][0]

        @tf.function
        def _behavior(s, id=0):
            b_ = behavior[id]
            _dist = b_.call_dist(s)[-1]
            affine_params = _dist.bijector.bijectors[-1]
            _mean = affine_params.shift
            _std = affine_params.scale
            _mean = tf.tile(_mean, [self._n_action_samples, 1])
            _std = tf.tile(_std, [self._n_action_samples, 1])
            _std = _std / (tf.reduce_max(_std, axis=-1, keepdims=True) / sigma)
            _new_dist = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=0.0, scale=1.0),
                bijector=tfp.bijectors.Chain(
                    _dist[0].bijector.bijectors[:-1]
                    + [tfp.bijectors.AffineScalar(shift=_mean, scale=_std), ]
                ),
                event_shape=[tf.shape(_mean)[-1]],
                batch_shape=[tf.shape(_mean)[0]]
            )
            a_n = _new_dist.sample()
            s_n = tf.tile(s, [self._n_action_samples, 1])
            q_n = q_fn(s_n, a_n)
            q_n = tf.reshape(q_n, (self._n_action_samples, s.shape[0]))  # shape: (n_sample, batch_size)
            a_indices = tf.argmax(q_n, axis=0)
            gather_indices = tf.stack(
                [a_indices, tf.range(s.shape[0], dtype=tf.int64)], axis=-1
            )
            actions = tf.reshape(a_n, (self._n_action_samples, s.shape[0], -1))  # shape: (n_sample, batch_size, a_dim)
            action = tf.gather_nd(actions, gather_indices)
            return action

        return _behavior

    def _get_behavior_sample(self, behavior, sigma):
        """Scaling the stddev of the behavior"""
        tfd = tfp.distributions
        if sigma is not None:
            sigma = tf.constant(sigma, dtype=tf.float32)

        @tf.function
        def _behavior(s, id=0):
            b_ = behavior[id]
            _dist = b_.call_dist(s)[-1]
            affine_params = _dist.bijector.bijectors[-1]
            _mean = affine_params.shift
            _std = affine_params.scale
            if sigma is not None:
                _std = _std / (tf.reduce_max(_std, axis=-1, keepdims=True) / sigma)
            _new_dist = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=0.0, scale=1.0),
                bijector=tfp.bijectors.Chain(
                    _dist[0].bijector.bijectors[:-1]
                    + [tfp.bijectors.AffineScalar(shift=_mean, scale=_std), ]
                ),
                event_shape=[tf.shape(_mean)[-1]],
                batch_shape=[tf.shape(_mean)[0]]
            )

            return _new_dist.sample()

        return _behavior

    def _get_dynamic(self, dynamic):
        @tf.function
        def _dynamic(s, a, id=0):
            s1 = []
            for d_ in dynamic:
                s1.append(d_(s, a)[0])
            # s1 = tf.reduce_mean(s1, axis=0)
            return s1[id][:, :-1], tf.reduce_mean(s1, axis=0)[:, -1]

        return _dynamic

    def _get_dynamic_diff(self, dynamic):
        """Computing the diff between pairs of the outputs"""

        @tf.function
        def _dynamic(s, a, id=0):
            s1 = []
            diff = []
            for i, d_ in enumerate(dynamic):
                s1_ = d_(s, a)[0]
                if i > 0:
                    diff += [sx - s1_ for sx in s1]
                s1.append(s1_)
            penalty = tf.reduce_max(tf.linalg.norm(diff, axis=-1), axis=0)
            # s1 = tf.reduce_mean(s1, axis=0)
            return s1[id][:, :-1], tf.reduce_mean(s1, axis=0)[:, -1], penalty

        return _dynamic

    def _compute_uncertainty_threshold(self):
        """Computing the uncertainty threshold with the real data set"""
        all_data = self._train_data.get_all_data()
        all_s1_np = all_data[0].numpy()
        all_a1_np = all_data[2].numpy()
        penalty = []
        num = 5000
        s1_split = np.array_split(all_s1_np, num)
        a1_split = np.array_split(all_a1_np, num)
        for i in range(num):
            penalty.append(self.dynamic_uncertainty(s1_split[i], a1_split[i])[-1])
        penalty = np.concatenate(penalty)
        self.d_uncertainty_threshold = np.percentile(penalty, self._uncertainty_percentile)
        print(f'The dynamics uncertainty threshold that computed from the data set is {self.d_uncertainty_threshold}')

    def _clear_sol(self):
        """ clear prev sol"""
        self.prev_sol = None

    def _init_sol(self, curr_x):
        """Initialize the actions using the behavior policy"""
        curr_x = np.expand_dims(curr_x, 0)
        repeat_curr_x = curr_x.repeat(self.pop_size, axis=0)
        actions = self.behavior(repeat_curr_x)
        mean = np.mean(actions, axis=0)
        mean = np.tile(mean, self.pred_len)
        mean = mean.reshape(self.pred_len, self._a_dim)

        return mean

    # MOPP(filtrating the trajectories using the stddev of the dynamics)
    def _mopp_bc_sol(self, observation, state=()):
        """ calculate the optimal inputs"""
        if len(observation.shape) == 2:
            assert observation.shape[0] == 1
            curr_x = observation[0]
        else:
            assert len(observation.shape) == 1
            curr_x = observation
        # get prev_sol
        if self.prev_sol is None:
            self.prev_sol = self._init_sol(curr_x)
        # get noised inputs
        noise = np.random.normal(
            loc=0, scale=1.0, size=(self.pop_size, self.pred_len,
                                    self._a_dim)) * self.noise_sigma
        noise = noise.astype('float32')
        noised_inputs = noise.copy()
        s = np.tile(curr_x, (self.pop_size, 1))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        rewards = []
        penalty = []
        if self.d_uncertainty_threshold is None:
            self._compute_uncertainty_threshold()
        for t in range(self.pred_len):
            # a_sample = self.behavior(s, np.random.randint(0, len(self._b_fns))) + noise[:, t, :]
            a_sample = self.behavior_maxq(s, np.random.randint(0, len(self._b_list)))
            a_sample_mix = self.beta * self.prev_sol[t, :] + (1 - self.beta) * a_sample.numpy()
            noised_inputs[:, t, :] = a_sample_mix
            s, r, penalty_ = self.dynamic_uncertainty(s,
                                                      tf.convert_to_tensor(a_sample_mix, dtype=tf.float32),
                                                      np.random.randint(0, len(self._d_list)))
            penalty.append(penalty_.numpy())
            rewards.append(r.numpy())
        # filtrating the trajectories
        penalty = np.array(penalty)  # shape: (pred_len, pop_size)
        penalty_label = np.where(penalty > self.d_uncertainty_threshold, penalty, 0)  # shape: (pred_len, pop_size)
        penalty_label_sum = np.sum(penalty_label, axis=0)
        select_ids = np.where(penalty_label_sum == 0)[0]
        if len(select_ids) < 0.2 * self.pop_size:
            # choose the top trajectories that with low penalty
            traj_penalty = np.sum(penalty, axis=0)
            s_thres = np.percentile(traj_penalty, 20)
            select_ids = np.where(traj_penalty <= s_thres)[0]
            rewards = np.array(rewards) - self._penalty_lambda * penalty_label
            # print('All the trajectories exceed the uncertainty threshold!')
        # clip actions
        noised_inputs = noised_inputs[select_ids]
        noised_inputs = np.clip(
            noised_inputs, self._a_min, self._a_max)
        # calc reward
        if self._use_value_fn:
            s = tf.tile(s, [self._n_action_samples, 1])
            a_last = self._behavior_sample(s, np.random.randint(0, len(self._b_list)))
            q_last = self.value_fn(s, a_last).numpy()
            q_last = np.reshape(q_last, [self._n_action_samples, -1])
            v_last = np.mean(q_last, axis=0)[np.newaxis, :]
            rewards = np.concatenate([np.array(rewards), v_last], axis=0)  # shape: (pred_len+1, pop_size)
            # a_last = self.behavior_maxq(s, np.random.randint(0, len(self._b_list)))
            # q_last = self.value_fn(s, a_last).numpy()
            # q_last = q_last[np.newaxis, :]
            # rewards = np.concatenate([np.array(rewards), q_last], axis=0)  # shape: (pred_len+1, pop_size)
        rewards = np.array(rewards)[:, select_ids]  # shape: (pred_len, pop_size)
        rewards = np.sum(rewards, axis=0)
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
    """Tensorflow module for MOPP-BC agent."""

    def _build_modules(self):
        n_b_fns = self._modules.n_b_fns
        self._b_nets = []
        for _ in range(n_b_fns):
            self._b_nets.append(self._modules.b_net_factory())
        self._d_nets = []
        self._q_nets = []
        n_q_fns = self._modules.n_q_fns
        for _rank in self._modules.d_out_ranks:
            self._d_nets.append(
                self._modules.d_net_factory(_rank)
            )
        for _ in range(n_q_fns):
            self._q_nets.append(
                [self._modules.q_net_factory(),
                 self._modules.q_net_factory(), ]  # source and target
            )
        self._b_alpha_entropy_var = tf.Variable(1.0)
        self._d_alpha_entropy_var = tf.Variable(1.0)

    def get_b_alpha_entropy(self):
        return utils.relu_v2(self._b_alpha_entropy_var)

    def get_d_alpha_entropy(self):
        return utils.relu_v2(self._d_alpha_entropy_var)

    def assign_alpha_entropy(self, b_alpha, d_alpha):
        self._b_alpha_entropy_var.assign(b_alpha)
        self._d_alpha_entropy_var.assign(d_alpha)

    @property
    def d_ae_variables(self):
        return [self._d_alpha_entropy_var]

    @property
    def b_ae_variables(self):
        return [self._b_alpha_entropy_var]

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
    def q_source_weights(self):
        q_weights = []
        for q_net, _ in self._q_nets:
            q_weights += q_net.weights
        return q_weights

    @property
    def q_target_weights(self):
        q_weights = []
        for _, q_net in self._q_nets:
            q_weights += q_net.weights
        return q_weights

    @property
    def q_source_variables(self):
        vars_ = []
        for q_net, _ in self._q_nets:
            vars_ += q_net.trainable_variables
        return tuple(vars_)

    @property
    def q_target_variables(self):
        vars_ = []
        for _, q_net in self._q_nets:
            vars_ += q_net.trainable_variables
        return tuple(vars_)


def get_modules(model_params, action_spec, state_spec):
    """Get agent modules."""
    model_params, _, d_out_ranks = model_params
    if d_out_ranks is None:
        d_out_ranks = [np.arange(state_spec.shape[0] + 1),
                       np.arange(state_spec.shape[0] + 1)[::-1]]
    if len(model_params) == 1:
        model_params = tuple([model_params[0]] * 2)

    def b_net_factory():
        return networks.ActorNetwork(
            action_spec,
            fc_layer_params=model_params[0][0])

    def d_net_factory(out_rank):
        return networks.ADMDynamic(
            state_dim=state_spec.shape[0],
            fc_layer_params=model_params[1],
            out_reranking=out_rank)

    def q_net_factory():
        return networks.CriticNetwork(
            fc_layer_params=model_params[2][0])

    modules = utils.Flags(
        b_net_factory=b_net_factory,
        d_net_factory=d_net_factory,
        q_net_factory=q_net_factory,
        d_out_ranks=d_out_ranks,
        n_b_fns=model_params[0][1],
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
