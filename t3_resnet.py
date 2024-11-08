import pr_core as core

from tensorflow import keras

from tframe.models.classifier import Classifier
from tframe.nets.image2image.unet import UNet
from tframe.nets.image2scalar.resnet import ResNet
from tframe.nets.net import Net


from tframe import console
from pr_core import th
from tframe.nets.net import Net
from tframe.models.model import Model
from tframe.quantity import *






# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'resnet'
def model(th):
  net = Net(name=model_name)
  # net.add(keras.layers.Conv2D(th.filters, 7, 2, padding='same'))
  # net.add(keras.layers.MaxPooling2D())
  net.add(ResNet(th.filters, th.kernel_size, th.archi_string))

  if th.dense_string != 'x':
    for neuron_num in th.dense_string.split('-'):
      net.add(keras.layers.Dense(int(neuron_num), use_bias=True))

  net.add(keras.layers.Dense(2, use_bias=True))

  model = Classifier(CrossEntropy(), [Accraucy(), MSE(), CrossEntropy()],
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
  th.boost = False
  th.radius = 80
  th.win_size = 512
  th.win_num=1

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
