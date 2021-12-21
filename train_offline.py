
"""Offline training pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# mujoco
try:
    from local_debug_logger import local_trace
except ImportError:
    local_trace = lambda: None

import os

from absl import app
from absl import flags
from absl import logging


import gin
import tensorflow as tf
import numpy as np

import agents
from rl_planning import train_eval_planning
from scripts import utils, path

tf.compat.v1.enable_v2_behavior()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Flags for which data to load.
flags.DEFINE_string('data_root_dir', 'data', 'Root directory for data.')
flags.DEFINE_string('benchmark_name', 'd4rl', 'benchmark name(d4rl/rl_unplugged/brac).')
flags.DEFINE_string('data_file_source', 'mujoco', 'data source(mujoco/adroit/pybullet).')
flags.DEFINE_string('data_file_name', 'hopper_mixed-v2', 'definete data file name(hdf5).')

# Flags for offline training.
flags.DEFINE_string('root_dir', 'learn_log',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('sub_dir', '0', '')
flags.DEFINE_enum('identifier', 'model_free',
                  ['imitation', 'model_free', 'model_based', 'constraints', 'ope', 'planning', 'pretrain'], '')
flags.DEFINE_string('agent_name', 'bcq', 'agent name.')
flags.DEFINE_string('env_name', 'Hopper-v2', 'env name.')
flags.DEFINE_integer('seed', 0, 'random seed, mainly for training samples.')
flags.DEFINE_integer('total_train_steps', int(5e5), '')
flags.DEFINE_integer('n_eval_episodes', 20, '')
# flags.DEFINE_integer('n_train', int(2e6), '')
# flags.DEFINE_integer('model_arch', 0, '')
# flags.DEFINE_integer('opt_params', 0, '')
flags.DEFINE_integer('save_freq', 5000, '')
flags.DEFINE_integer('num_transitions', -1, 'How many transitions used for training, -1 means all the data')
flags.DEFINE_boolean('state_normalize', False, '')
flags.DEFINE_boolean('reward_normalize', True, '')
flags.DEFINE_boolean('score_normalize', True, '')
flags.DEFINE_boolean('env_evaluate', False, '')
flags.DEFINE_multi_string('b_ckpt', 'placeholder', 'behavior model checkpoint name, e.g., b_gaussian')
# dynamics (model-based rl or ope)
flags.DEFINE_string('f_ckpt', 'placeholder', 'dynamic model checkpoint name')
flags.DEFINE_integer('fic_replay_buffer_size', int(1e6), 'fictitious rollouts nums')
flags.DEFINE_integer('batch_size', 256, '')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  data_dir = os.path.join(
      FLAGS.data_root_dir,
      FLAGS.benchmark_name,
      FLAGS.data_file_source,
      FLAGS.data_file_name,
      )

  data_file = path.abs_file_path(__file__, data_dir)

  # Setup summary_log dir & model_save dir.
  if FLAGS.sub_dir == 'auto':
    sub_dir = utils.get_datetime()
  else:
    sub_dir = FLAGS.sub_dir
  log_dir = os.path.join(
      FLAGS.root_dir,
      FLAGS.env_name,
      FLAGS.data_file_source+'_'+FLAGS.data_file_name,
      str(FLAGS.num_transitions)+'_'+str(FLAGS.state_normalize),
      FLAGS.agent_name,
      sub_dir,
      str(FLAGS.seed),
      )
  model_dir = os.path.join(
      'learn_model',
      FLAGS.env_name,
      FLAGS.data_file_name,
      str(FLAGS.num_transitions)+'_'+str(FLAGS.state_normalize),
      )
  utils.maybe_makedirs(log_dir)
  utils.maybe_makedirs(model_dir)

  if FLAGS.identifier == 'planning':
    # behavior, dynamic; b_out_ranks; d_out_ranks
    model_arch = ((((300, 300), (300, 100)),
                  ((300, 300), (300, 100))), None, None)
    opt_params = (('adam', 1e-3), ('adam', 1e-3), ('adam', 1e-3), ('adam', 1e-3), ('adam', 1e-3))
  else:
    raise ValueError("Wrong identifier!")

  if FLAGS.identifier == 'planning':
    eval_results = train_eval_planning.train_eval_planning(
        log_dir=log_dir,
        model_dir=model_dir,
        benchmark_name=FLAGS.benchmark_name,
        data_file_source=FLAGS.data_file_source,
        data_file=data_file,
        agent_module=agents.AGENT_MODULES_DICT[FLAGS.agent_name],
        env_name=FLAGS.env_name,
        total_train_steps=FLAGS.total_train_steps,
        n_eval_episodes=FLAGS.n_eval_episodes,
        optimizers=opt_params,
        num_transitions=FLAGS.num_transitions,
        behavior_ckpt_name=FLAGS.b_ckpt[0],
        dynamics_ckpt_name=FLAGS.f_ckpt,
        reward_normalize=FLAGS.reward_normalize,
        state_normalize=FLAGS.state_normalize,
        score_normalize=FLAGS.score_normalize,
        evaluate=FLAGS.env_evaluate,
    )
  else:
    raise ValueError("Wrong identifier!")

  results_file = os.path.join(log_dir, 'results_reward.npy')
  with tf.io.gfile.GFile(results_file, 'w') as f:
    np.save(f, eval_results)


if __name__ == '__main__':
  app.run(main)
