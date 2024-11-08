import prgan_core as core

from tensorflow import keras

from pr.architectures.esrgan import ESRGAN
from pr.architectures.half_unet import HalfUnet
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
th.task_name = 'UESRGan'

def generator():
  net = Net('Generator')
  # net.add(UNet(name='UNet', link_indices=th.bridges, use_batchnorm=False))
  # net.add(keras.layers.Conv2D(1,1, activation='sigmoid', use_bias=False))
  # model = Model(GlobalBalancedError(), [PSNR(max_val=1.0), SSIM(max_val=1.0), MAE()],
  #               net, name='UNet')
  # model = Model(WMAE(), [MSE(), MAE()],
  #               net, name=model_name)
  net.add(HalfUnet(output_height=2))
  net.add(keras.layers.Conv2D(1,1, activation='sigmoid', use_bias=False))
  net.add(ESRGAN())
  model = Model(GlobalBalancedError(), [PSNR(max_val=1.0), SSIM(max_val=1.0), MAE()],
                net, name='Generator')
  model.mark = 'generator'
  return model

def discriminator():
  net = Net('Resnet')
  net.add(ResNet(3, 3, '1-2-2-2-1'))

  net.add(keras.layers.Dense(2, use_bias=True))

  model = Classifier(CrossEntropy(), [Accraucy(), MSE(), CrossEntropy()],
                     net, name='Resnet')
  model.mark = 'discriminator'
  return model

th.generator = generator
th.discriminator = discriminator

def main(_):
  console.start('{} on PR task'.format('GAN'))

  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.feature_type = 1
  th.target_type = 1
  th.boost = False
  th.radius = 80
  th.win_size = None
  th.win_num=1

  # th.train_indices = '1,2,11,13'
  # th.val_indices = '1,2,11,13'
  # th.test_indices = '1,2,11,13'
  # th.train_samples = '3t3'
  # th.val_samples = '3t3'
  # th.test_samples = '3t3'

  th.train_indices = '5,23'
  th.val_indices = '5,23'
  th.test_indices = '5,23'
  th.train_samples = 'bead'
  th.val_samples = 'bead'
  th.test_samples = 'bead'

  # th.train_indices = '5'
  # th.val_indices = '5'
  # th.test_indices = '5'
  # th.train_samples = 'bead'
  # th.val_samples = 'bead'
  # th.test_samples = 'bead'

  th.train_set_ratio = ':0.6'
  th.val_set_ratio = '0.6:0.8'
  th.test_set_ratio = '0.8:'
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
  th.batch_size = 1
  th.learning_rate = 0.0001
  th.patience = 100
  th.val_batch_size = 1
  th.save_model = True
  th.load_model = False
  th.input_shape = (1024, 1280, 1)
  th.non_train_input_shape = (1024, 1280, 1)
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
