import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
from PIL import Image
import numpy as np
import cv2
import math
import torch
import sys
import datetime
import logging
from collections import defaultdict

import config as c


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)



class dataset_(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.img_filenames = list(sorted(os.listdir(img_dir)))
        self.transform = transform
        self.totensor = T.ToTensor()
    
    def __len__(self):
        return len(self.img_filenames)
    
    def __getitem__(self, index):
        img_paths = os.path.join(self.img_dir, self.img_filenames[index])
        img = Image.open(img_paths).convert("RGB")
        img = self.transform(img)
        return img
    
    
transform_train = T.Compose([
    T.RandomCrop(c.crop_size_train),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    # T.RandomCrop(c.crop_size_train),
    T.ToTensor()
])


transform_val = T.Compose([
    # T.CenterCrop(c.crop_size_train),
    T.Resize([c.resize_size_test, c.resize_size_test]),
    T.ToTensor(),
])


def load_dataset(train_data_dir, test_data_dir, batchsize_train, batchsize_test, generated_cover_image_dir=None):

    train_loader = DataLoader(
        dataset_(train_data_dir, transform_train),
        batch_size=batchsize_train,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset_(test_data_dir, transform_val),
        batch_size=batchsize_test,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=True
    )

    if generated_cover_image_dir == None:
        return train_loader, test_loader
            
    else:
        generated_cover_loader = DataLoader(  # for testing
            dataset_(generated_cover_image_dir, transform_val),
            batch_size=batchsize_test,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
            drop_last=True
        )

        return train_loader, test_loader, generated_cover_loader
    


def quantization(tensor):
    return torch.round(torch.clamp(tensor*255, min=0., max=255.))/255

# def quantization_v2(tensor):
#     return torch.round(255 * (tensor - tensor.min()) / (tensor.max() - tensor.min()))/255

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def Fibonacci(n):
   if n == 1 or n == 2:
       return 1
   else:
       return(Fibonacci(n-1) + Fibonacci(n-2))
   

def cat_map(tensor, p=1, q=1, epoch=10, obfuscate=True):
    '''
    key k is composed by p, q and epoch
    tensor: batch x c x h x w
    '''
    matric = torch.tensor([[Fibonacci(2*epoch-1), Fibonacci(2*epoch)], [Fibonacci(2*epoch), Fibonacci(2*epoch+1)]])
    matric_inverse = torch.tensor([[Fibonacci(2*epoch+1), -1*Fibonacci(2*epoch)], [-1*Fibonacci(2*epoch), Fibonacci(2*epoch-1)]])
    # print(torch.mm(matric, matric_inverse))
    
    if obfuscate != True: # restore pert
        matric = matric_inverse

    h = tensor.shape[2]; w = tensor.shape[3]

    processed_tensor = torch.zeros_like(tensor)
    for x in range(0, h):
        for y in range(0, w):
            new_x, new_y = torch.mm(matric, torch.tensor(([x], [y]))) % h
            processed_tensor[:, :, new_x.item(), new_y.item()] = tensor[:, :, x, y]
    
    return processed_tensor



def calculate_rmse(img1, img2):
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    rmse = np.sqrt(mse)

    return np.mean(rmse)


def calculate_rmses(t1, t2):
    rmes_list = []
    for i in range(t1.shape[0]):
        rmes_list.append(calculate_rmse(t1[i], t2[i]))
    return np.mean(np.array(rmes_list))


def calculate_mae(img1, img2):

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    apd = np.mean(np.abs(img1 - img2))
    if apd == 0:
        return float('inf')

    return np.mean(apd)


def calculate_maes(t1, t2):
    mae_list = []
    for i in range(t1.shape[0]):
        mae_list.append(calculate_mae(t1[i], t2[i]))
    return np.mean(np.array(mae_list))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_psnrs(t1, t2):
    psnr_list = []
    for i in range(t1.shape[0]):
        psnr_list.append(calculate_psnr(t1[i], t2[i]))
    return np.mean(np.array(psnr_list))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssims(t1, t2):
    ssim_list = []
    for i in range(t1.shape[0]):
        ssim_list.append(calculate_ssim(t1[i], t2[i]))
    return np.mean(np.array(ssim_list))


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = img1.transpose((1, 2, 0))
    img2 = img2.transpose((1, 2, 0))

    # print(img1.shape)
    # print(img2.shape)
    # bk
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    





'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# --------------------------------------------
# logger
# --------------------------------------------
'''

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logging.Formatter.converter = beijing



def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


'''
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass



class MetricMonitor:
    '''
    MetricMonitor helps to track metrics such as accuracy or loss during training and validation and shows them on terminal.
     '''
    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

