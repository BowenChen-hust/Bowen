import numpy as np

from pr.pr_agent import PRAgent
from pr.pr_set import PhaseSet

from typing import List


def load_data(th, data_dir) -> [PhaseSet, PhaseSet, PhaseSet]:

  # Load datasets normally
  datasets = PRAgent.load_data(
    data_dir, th)

  # Apply configs and report detail
  for i, config in enumerate((th.train_config, th.val_config, th.test_config)):
    # Apply config if required
    if config not in (None, '', 'x', '-'):
      datasets[i] = datasets[i].get_subset_by_config_str(config)
    # Report detail
    datasets[i].report()

  # Return datasets
  train_set, val_set, test_set = datasets
  return train_set, val_set, test_set




if __name__ == '__main__':
  from pr.pr_configs import PRConfig
  th = PRConfig()

  th.train_indices = '1-2'
  th.val_indices = '1-2'
  th.test_indices = '5'
  th.train_samples = '3t3'
  th.val_samples = '3t3'
  th.test_samples = 'bead'
  th.train_set_ratio = ':0.7'
  th.val_set_ratio = '0.7:'
  th.test_set_ratio = None
  th.radius = 80
  th.truncate_at = 12
  th.win_size = 512
  th.win_num=4
  # th.fn_pattern = '01-'
  # th.fn_pattern = '*62-'

  th.feature_type = 1

  # th.train_indices = '4'
  # th.val_indices = '4'
  # th.test_indices = '4'
  # th.train_config = ':10'
  # th.val_config = '10:15'
  # th.test_config = '15:'

  # th.pr_dev_code = 'dev.0'
  # th.prior_size = 12
  # th.use_prior = True

  train_set, val_set, test_set = load_data(th, r'D:/learning/lambai_v2-main/01-PR/data')
  assert isinstance(train_set, PhaseSet)
  assert isinstance(val_set, PhaseSet)
  assert isinstance(test_set, PhaseSet)
  for batch in train_set.gen_batches(batch_size=16, updates_per_round=10, is_training=True,
                                     shuffle=True):
    batch.view()
  # test_set.view_aberration()
  # train_set.view()
  # test_set.view()

  # win_num = 10
  # win_size = 512
  # test_set.test_window(win_size, win_num)

