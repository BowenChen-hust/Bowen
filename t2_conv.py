import pr_core as core

from tensorflow import keras
from pr.architectures.convnet import ConvNet
from tframe.nets.net import Net


from tframe import console
from pr_core import th
from tframe.nets.net import Net
from tframe.models.model import Model
from tframe.quantity import *






# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'ConvNet'
def model():
  net = Net(model_name)
  net.add(ConvNet(name=model_name))
  net.add(keras.layers.Conv2D(1,1, activation='sigmoid', use_bias=False))
  model = Model(GlobalBalancedError(), [GlobalBalancedError(), MSE(),
                                        PSNR(max_val=1.0),
                                        SSIM(max_val=1.0)
                                        ],
                net, name=model_name)
  return model

th.model = model
def main(_):
  console.start('{} on PR task'.format(model_name))

  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.feature_type = 2
  th.target_type = 1
  th.boost = True
  th.radius = 80
  th.win_size = None
  th.win_num=4

  th.train_indices = '05'
  th.val_indices = '05'
  th.test_indices = '01'
  th.train_samples = 'bead'
  th.val_samples = 'bead'
  th.test_samples = '3t3'

  # th.train_indices = '1,2'
  # th.val_indices = '1,2'
  # th.test_indices = '5'
  # th.train_samples = '3t3'
  # th.val_samples = '3t3'
  # th.test_samples = 'bead'

  th.train_set_ratio = ':0.7'
  th.val_set_ratio = '0.7:'
  th.test_set_ratio = ':'
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
  th.epoch = 100000
  th.batch_size = 16
  th.learning_rate = 0.0001
  th.patience = 30
  th.val_batch_size = 8
  th.save_model = True
  th.load_model = False
  th.input_shape = (None, None, 2)
  th.non_train_input_shape = (512, 512, 2)
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
