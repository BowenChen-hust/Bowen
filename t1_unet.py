import pr_core as core

from tensorflow import keras
from tframe.nets.image2image.unet import UNet
from tframe.nets.net import Net


from tframe import console
from pr_core import th
from tframe.nets.net import Net
from tframe.models.model import Model
from tframe.quantity import *






# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
th.task_name = 'UNet'
def model():
  net = Net('UNet')
  net.add(UNet(name='UNet'))
  net.add(keras.layers.Conv2D(1,1, activation='sigmoid', use_bias=False))
  model = Model(GlobalBalancedError(), [PSNR(max_val=1.0), SSIM(max_val=1.0)],
                     net, name='UNet')
  # model = Model(WMAE(), [MSE(), MAE()],
  #               net, name=model_name)
  return model

th.model = model
def main(_):
  console.start('{} on PR task'.format('UNet'))

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
  # th.train_indices = '5'
  # th.val_indices = '5'
  # th.test_indices = '5'
  # th.train_samples = 'bead'
  # th.val_samples = 'bead'
  # th.test_samples = 'bead'

  #3t3
  # th.train_indices = '02'
  # th.val_indices = '02'
  # th.test_indices = '02'
  # th.train_samples = '3t3'
  # th.val_samples = '3t3'
  # th.test_samples = '3t3'

  # hela
  # th.train_indices = '03'
  # th.val_indices = '03'
  # th.test_indices = '03'
  # th.train_samples = 'Hek'
  # th.val_samples = 'Hek'
  # th.test_samples = 'Hek'

  # rbc
  th.train_indices = '04'
  th.val_indices = '04'
  th.test_indices = '04'
  th.train_samples = 'rbc'
  th.val_samples = 'rbc'
  th.test_samples = 'rbc'

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
  th.batch_size = 16 #16
  th.learning_rate = 0.0001
  th.patience = 1000
  th.val_batch_size = 8  #8
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
