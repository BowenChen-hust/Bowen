import pr_core as core
from pr_core import th

import tensorflow.keras as keras

from pr.architectures.esrgan import ESRGAN
from pr.architectures.half_unet import HalfUnet
from tframe.models.classifier import Classifier
from tframe.nets.image2image.unet import UNet
from tframe.nets.image2scalar.resnet import ResNet
from tframe.nets.net import Net


from tframe import console
from tframe.nets.net import Net
from tframe.models.model import Model
from tframe.quantity import *
from pr.architectures.esrgan import PixelShuffle


# class generator_loss(Quantity):
#   def __init__(self):
#     super(generator_loss, self).__init__('genloss', True)
#
#   def __call__(self, fake_output):
#     return self.function(fake_output)
#
#   def function(self, fake_output):
#     return -tf.reduce_mean(fake_output)
#
#
# class disc_real_loss(Quantity):
#   def __init__(self):
#     super(disc_real_loss, self).__init__('dis_real_loss', True)
#
#   def __call__(self, real_output):
#     return self.function(real_output)
#
#   def function(self, real_output):
#     real_loss = -tf.reduce_mean(real_output)
#     return real_loss
#
# class disc_fake_loss(Quantity):
#   def __init__(self):
#     super(disc_fake_loss, self).__init__('dis_fake_loss', True)
#
#   def __call__(self,  fake_output):
#     return self.function(fake_output)
#
#   def function(self, fake_output):
#     fake_loss = tf.reduce_mean(fake_output)
#     return fake_loss



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
th.task_name = 'Autoencoder'

def generator():
  net = Net('Generator')
  net.add(HalfUnet(output_height=2, link_indices=[]))
  net.add(PixelShuffle(upscale=4))
  # net.add(UNet(activation=keras.layers.LeakyReLU()))
  model = Model(MAE(), [PSNR(max_val=1.0), SSIM(max_val=1.0)], net, name='Generator')
  model.mark = 'generator'
  return model

# def discriminator():
#   net = Net('Resnet')
#   net.add(ResNet(16, 3, '1-2-2-1', activation=keras.layers.LeakyReLU(0.2),
#                  bn=False, use_bias=False))
#   #
#   #
#   # filter = 16
#   # for _ in range(2):
#   #   net.add(keras.layers.Conv2D(filter, 3, 1))
#   #   net.add(keras.layers.BatchNormalization())
#   #   net.add(keras.layers.LeakyReLU())
#   #
#   # for i in range(4):
#   #   filter *= 2
#   #   net.add(keras.layers.Conv2D(filter, 3, 2))
#   #   net.add(keras.layers.BatchNormalization())
#   #   net.add(keras.layers.LeakyReLU())
#   #   for _ in range(2):
#   #     net.add(keras.layers.Conv2D(filter, 3, 1))
#   #     net.add(keras.layers.BatchNormalization())
#   #     net.add(keras.layers.LeakyReLU())
#
#   net.add(keras.layers.Flatten())
#   net.add(keras.layers.Dropout(rate=0.5))
#   net.add(keras.layers.Dense(1, use_bias=False))
#
#   model = Classifier([disc_real_loss(), disc_fake_loss()], None,
#                      net, name='Resnet')
#   model.mark = 'discriminator'
#   return model

th.model = generator
# th.discriminator = discriminator

def main(_):
  console.start('{} on PR task'.format('UPS'))

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

  # 3t3
  th.train_indices = '19,20'
  th.val_indices = '19,20'
  th.test_indices = '19,20'
  th.train_samples = '3t3'
  th.val_samples = '3t3'
  th.test_samples = '3t3'
  th.train_set_ratio = ':0.6'
  th.val_set_ratio = '0.6:0.8'
  th.test_set_ratio = '0.8:'

  # hela
  # th.train_indices = '30'
  # th.val_indices = '30'
  # th.test_indices = '30'
  # th.train_samples = 'hela'
  # th.val_samples = 'hela'
  # th.test_samples = 'hela'

  # rbc 68 + 8 + 9 = 85
  # th.train_indices = '17'
  # th.val_indices = '17'
  # th.test_indices = '17'
  # th.train_samples = 'rbc'
  # th.val_samples = 'rbc'
  # th.test_samples = 'rbc'

  # wbc b 54 + 18 + 18 = 90
  # th.train_indices = '34'
  # th.val_indices = '34'
  # th.test_indices = '34'
  # th.train_samples = 'wbc_b'
  # th.val_samples = 'wbc_b'
  # th.test_samples = 'wbc_b'

  # th.train_set_ratio = ':0.8'
  # th.val_set_ratio = '0.8:0.9'
  # th.test_set_ratio = '0.9:'
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
  th.patience = 100
  th.val_batch_size = 16
  th.save_model = True
  th.load_model = False
  # th.input_shape = (1024, 1280, 1)
  # th.non_train_input_shape = (1024, 1280, 1)
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
