import pr_core as core

from tensorflow import keras
from tframe.nets.net import Net
from pr.architectures.half_unet import HalfUnet

from tframe import console
from pr_core import th
from tframe.models.model import Model
from tframe.quantity import *






# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
th.task_name = 'HalfUnet'
def model():
  net = Net('HalfUnet')
  net.add(HalfUnet(name='HalfUnet'))
  net.add(keras.layers.Conv2D(16, 3, padding='same', activation='relu')) 
  net.add(keras.layers.Conv2D(1,1, activation='sigmoid', use_bias=False))
  model = Model(GlobalBalancedError(), [PSNR(max_val=1.0), SSIM(max_val=1.0)],
                     net, name='HalfUnet')
  # model = Model(WMAE(), [MSE(), MAE()],
  #               net, name=model_name)
  return model

th.model = model
def main(_):
  console.start('{} on PR task'.format('HalfUnet'))

  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.feature_type = 1
  th.target_type = 1
  th.boost = False
  th.radius = 80
  th.win_size = 512
  th.win_num=1

  # bead
  # th.train_indices = '5,23'
  # th.val_indices = '5,23'
  # th.test_indices = '5,23'
  # th.train_samples = 'bead'
  # th.val_samples = 'bead'
  # th.test_samples = 'bead'

  #3t3
  th.train_indices = '01'
  th.val_indices = '01'
  th.test_indices = '01'
  th.train_samples = '3t3'
  th.val_samples = '3t3'
  th.test_samples = '3t3'
  # hela
  # th.train_indices = '30'
  # th.val_indices = '30'
  # th.test_indices = '30'
  # th.train_samples = 'hela'
  # th.val_samples = 'hela'
  # th.test_samples = 'hela'

  # rbc
  # th.train_indices = '17'
  # th.val_indices = '17'
  # th.test_indices = '17'
  # th.train_samples = 'rbc'
  # th.val_samples = 'rbc'
  # th.test_samples = 'rbc'

  # wbc b
  # th.train_indices = '34'
  # th.val_indices = '34'
  # th.test_indices = '34'
  # th.train_samples = 'wbc_b'
  # th.val_samples = 'wbc_b'
  # th.test_samples = 'wbc_b'

  th.train_set_ratio = ':0.8'
  th.val_set_ratio = '0.8:0.9'
  th.test_set_ratio = '0.9:'
  th.radius = 80
  th.truncate_at = 12

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  pass
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.bridges = '0,1,2,3'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100000  #100000
  th.batch_size = 2  #16
  th.learning_rate = 0.0001
  th.patience = 1000
  th.val_batch_size = 2  #8
  th.save_model = True
  th.load_model = False
  th.input_shape = (None, None, 1)
  th.non_train_input_shape = (512, 512, 1)
  th.overwrite = True
  th.random_ditto=False
  th.summary = True
  # th.trainer_setup('alpha')

  th.probe = True
  th.probe_cycle = 1

  th.rehearse = False
  th.train = True
  
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  main(None)
