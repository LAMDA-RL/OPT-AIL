from collections import defaultdict
import json
import os
import csv
import shutil
import wandb
import torch
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from termcolor import colored


COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('duration', 'D', 'time')
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float')
]


AGENT_TRAIN_FORMAT = {
    'sac': [
        # ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float')
    ],
    'dac': [
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('discriminator_loss', 'DLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('gradient_penalty', 'GP', 'float')
    ],
    'bc': [
        ('actor_loss', 'ALOSS', 'float'),
    ],
    'opt_ail': [
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('discriminator_loss', 'DLOSS', 'float'),
        ('optimistic_loss', 'OLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('gradient_penalty', 'GP', 'float')
    ],
    'mb_ail': [
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('discriminator_loss', 'DLOSS', 'float'),
        ('dynamics_loss', 'DYLOSS', 'float'),
        ('dynamics_holdout_loss', 'HLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('gradient_penalty', 'GP', 'float')
    ],
    'cmil': [
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('discriminator_loss', 'DLOSS', 'float'),
        ('dynamics_loss', 'DYLOSS', 'float'),
        ('dynamics_holdout_loss', 'HLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('gradient_penalty', 'GP', 'float'),
        ('unc_std', 'STD', 'float'),
    ],
    'mbpo': [
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('dynamics_loss', 'DYLOSS', 'float'),
        ('dynamics_holdout_loss', 'HLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('gradient_penalty', 'GP', 'float')
    ],
    'hyper': [
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('discriminator_loss', 'DLOSS', 'float'),
        ('dynamics_loss', 'DYLOSS', 'float'),
        ('dynamics_holdout_loss', 'HLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('gradient_penalty', 'GP', 'float')
    ],
}

class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return None
        data = None
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()
        return data


class Logger(object):
    def __init__(self,
                 log_dir,
                 save_tb=False,
                 log_frequency=10000,
                 agent='sac',
                 writer=None):
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        if writer:
            self._sw = writer
        else:
            if save_tb:
                tb_dir = os.path.join(log_dir, 'tb')
                if os.path.exists(tb_dir):
                    try:
                        shutil.rmtree(tb_dir)
                    except:
                        print("logger.py warning: Unable to remove tb directory")
                        pass
                self._sw = SummaryWriter(tb_dir)
            else:
                self._sw = None
        # each agent has specific output format for training
        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'),
                                     formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'),
                                    formating=COMMON_EVAL_FORMAT)

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def _try_wandb_log_dict(self, data, step, prefix):
        # Only log if wandb is installed and a run is active
        if wandb is None or data is None or wandb.run is None:
            return
        log_data = {}
        for key, value in data.items():
            if key == 'step':
                continue
            log_data[f'{prefix}/{key}'] = value
        if len(log_data) == 0:
            return
        try:
            wandb.log(log_data, step=step)
        except Exception as exc:
            print(f"logger.py warning: wandb log failed ({exc})")

    def log(self, key, value, step, n=1, log_frequency=1):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_video(self, key, frames, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step, save=True, ty=None):
        train_data = eval_data = None
        if ty is None:
            train_data = self._train_mg.dump(step, 'train', save)
            eval_data = self._eval_mg.dump(step, 'eval', save)
        elif ty == 'eval':
            eval_data = self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            train_data = self._train_mg.dump(step, 'train', save)
        else:
            raise f'invalid log type: {ty}'
        self._try_wandb_log_dict(train_data, step, 'train')
        self._try_wandb_log_dict(eval_data, step, 'eval')


# def setup_logger(filepath):
#     file_formatter = logging.Formatter(
#         "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
#         datefmt='%Y-%m-%d %H:%M:%S',
#     )
#     logger = logging.getLogger('example')
#     handler = logging.StreamHandler()
#     handler.setFormatter(file_formatter)
#     logger.addHandler(handler)

#     file_handle_name = "file"
#     if file_handle_name in [h.name for h in logger.handlers]:
#         return
#     if os.path.dirname(filepath) is not '':
#         if not os.path.isdir(os.path.dirname(filepath)):
#             os.makedirs(os.path.dirname(filepath))
#     file_handle = logging.FileHandler(filename=filepath, mode="a")
#     file_handle.set_name(file_handle_name)
#     file_handle.setFormatter(file_formatter)
#     logger.addHandler(file_handle)
#     logger.setLevel(logging.DEBUG)
#     return logger
