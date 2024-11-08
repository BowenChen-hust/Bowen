import sys, os
import numpy as np

DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)

sys.path.append(os.path.join(sys.path[0], 'roma'))
sys.path.append(os.path.join(sys.path[0], 'ditto'))
sys.path.append(os.path.join(sys.path[0], 'lambo'))

os.environ["CUDA_VISIBLE_DEVICES"]= "0" if not 'home' in sys.path[0] else "1"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if not 'home' in sys.path[0]:
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 3)]
  )
else:
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 25)]
  )

#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here
# DIR_DEPTH = 2
# ROOT = os.path.abspath(__file__)
# for _ in range(DIR_DEPTH):
#   ROOT = os.path.dirname(ROOT)
#   if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
# =============================================================================
import tframe
tframe.set_random_seed(404)
from tframe.core.agent import Agent
from tframe.trainers.gan_trainer import GANTrainer as Trainer
from pr.pr_configs import PRConfig

from tframe import console
th = PRConfig()

import pr_du as du
from pr.pr_util import probe


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
# th.allow_growth = False
# th.gpu_memory_fraction = 0.40

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.print_cycle = 1
th.epoch = 1000
th.batch_size = 32
th.learning_rate = 0.0003
th.updates_per_round = 30

th.validate_train_set = False
th.validate_val_set = True
th.validate_test_set = False
th.val_batch_size = -1
th.save_model = False
th.load_model = True
th.overwrite = False
th.patience = 30

th.probe_cycle = 10
th.train_probe_ids = '0'
th.val_probe_ids = '0'
th.test_probe_ids = '0'

# th.evaluate_train_set = True
# th.evaluate_val_set = True
# th.evaluate_test_set = True



def activate():

  # Build model
  assert callable(th.generator)
  assert callable(th.discriminator)
  generator = th.generator()
  discriminator = th.discriminator()
  generator.mark ='lr{}-bs{}-f{}-t{}-b{}-l{}-{}{}{}-{}{}{}-{}{}{}-v5'.format(th.learning_rate,
                                                                      th.batch_size,  th.feature_type, th.target_type, th.boost, generator.loss[0].name,
                                                                      th.train_indices, th.train_samples, th.train_set_ratio,
                                                                      th.val_indices, th.val_samples, th.val_set_ratio,
                                                                      th.test_indices, th.test_samples, th.test_set_ratio)
  for model in [generator, discriminator]:
    model.mark = model.mark.replace('True','T')
    model.mark = model.mark.replace('False','F')
    model.mark = model.mark.replace(':','t')

  agent = Agent(generator.mark, th.task_name, max_num_saved_models=4)
  agent.config_dir(__file__)

  print(agent.data_dir)
  if  'yiz/share/projects/lambai_v2/01-PR/data' in agent.data_dir:
    agent.data_dir = agent.data_dir.replace('yiz/share/projects/lambai_v2/01-PR/data', 'D:/learning/lambai_v2-main/01-PR/data')
    # agent.data_dir = '/home/data1/yi/prdata/data'

  print(agent.data_dir)

  # if th.rehearse:
  #   trainer = Trainer(model, agent, th, None, None, None)
  #   trainer.probe = lambda: probe(trainer)
  #   trainer.train()
  #   return

  # Load data
  train_set, val_set, test_set = du.load_data(th, agent.data_dir)


  # Train or evaluate
  if th.train:
    trainer = Trainer(generator, discriminator,
                      agent, th, train_set, val_set, test_set)
    trainer.probe =lambda : probe(trainer)
    trainer.train()
  else:
    # Evaluate on test set
    trainer = Trainer(generator, discriminator,
                      agent, th, train_set, val_set, test_set)
    model.keras_model, _ = trainer.agent.load_model(model.mark)
    model.evaluate(test_set)

  # End
  console.end()
