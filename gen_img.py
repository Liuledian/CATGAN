from dagan_trainer import DaganTrainer
from discriminator import Discriminator
from generator import Generator
from dataset import create_dagan_dataloader
from utils.parser import get_omniglot_dagan_args
import torchvision.transforms as transforms
import torch
import os
import torch.optim as optim
import numpy as np
import imageio
import torch
import time
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from PIL import Image
import PIL
import warnings


def render_img( arr,path):
    arr = (arr * 0.5) + 0.5
    arr = np.uint8(arr * 255)
    #new_img=Image.fromarray(arr, mode="L").transpose(PIL.Image.TRANSPOSE)
    new_img = Image.fromarray(arr, mode="L")
    new_img.save(path)


if __name__ == '__main__':
    pth_path='/data_c/hsy/image_generation/result/model_ckpt/g1.pth'
    img_path='1065_01.png'
    g = torch.load(pth_path)
    g=g.cuda()
    g.eval()

    mid_pixel_value=1.
    in_channels=1
    train_transform = transforms.Compose(
        [
            #transforms.ToPILImage(),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(
                (mid_pixel_value,) * in_channels, (mid_pixel_value,) * in_channels
            ),
        ]
    )

    img = Image.open(img_path).convert('L')
    img=train_transform(img).cuda()

    img=img.unsqueeze(0)
    z = torch.randn((1, g.z_dim)).cuda()
    #print(img.shape)
    #print(z.shape)
    gen_img=g(img, z)
    #print(gen_img.shape)
    render_img(torch.squeeze(img).cpu(),'1.png')
    render_img(torch.squeeze(gen_img).cpu().detach().numpy(), '2.png')





























