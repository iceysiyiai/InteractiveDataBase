import torch.nn as nn
from torchvision import transforms, utils
import pandas as pd
import random
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, utils
import numpy as np


IMAGENET_MU = [0.485, 0.456, 0.406]
IMAGENET_SIGMA = [0.229, 0.224, 0.225]


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())


class Clip(object):
    """Pytorch transformation that clips a tensor to be within [0,1]"""
    def __init__(self):
        return

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): tensor to be clipped.
        Returns:
            Tensor: clipped tensor.
        """
        t = tensor.clone()
        t[t > 1] = 1
        t[t < 0] = 0
        return t
    
def get_detransform(mu=IMAGENET_MU, sigma=IMAGENET_SIGMA):
    detransform = transforms.Compose([
        NormalizeInverse(mu, sigma),
        Clip(),
        # transforms.ToPILImage(),
    ])
    return detransform



def plot_image(tensor):
    plt.figure()
    plt.imshow(tensor.numpy().transpose(1, 2, 0))
    plt.show()

def load_random_image():
    # Generate random image
    n = random.randint(0, 250)
    print('Random number generated: '+ str(n))
    
    # Read from data set, split path and label
    dataset250 = pd.read_csv('val_250/path_lable.txt', sep = '/', header = None, names = ["path", "label"])
    img_name_label = dataset250.iloc[n, 1]
    path, label_str = img_name_label.split()
    label = int(label_str)
    
    # Print path and label
    print('Image path: {}'.format(path))
    print('Image label:{}'.format(label))
    
    # function to convert PIL images to tensors.
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()

    # Read the image from file.
    img_path = "val_250/val_250/"+ path
    image = Image.open(img_path)
    rgb_image = pil2tensor(image)
    plot_image(rgb_image)
    
    return [image, label]


def mask_transform(mask_input):
    mask_3d = mask_input.reshape(1, 224, 224)
    msk = torch.from_numpy(mask_3d).float()
    return msk

def img_transform(img_input):
    tensor_trans = transforms.ToTensor()
    img_tensor = tensor_trans(img_input)
    img = img_tensor.float()
    return img

def transform_preservation (img_input, mask_input):
    masked_img = mask_input * img_input
    return masked_img

def transform_deletion(img_input, mask_input):
    masked_img = (1-mask_input) * img_input
    return masked_img

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def transform_Gaussian(img_input, mask_input):
    blur_part = mask_input * img_input
    transform_g=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.ToTensor(),
                              AddGaussianNoise(0.15, 0.2)  
                           ])
    masked_img = img_input * (1-mask_input) + mask_input * transform_g(blur_part)
    return masked_img


def transform_Noise(img_input, mask_input):
    blur_part = mask_input * img_input
    transform_n = transforms.GaussianBlur(kernel_size=(17, 17), sigma=(0.1, 200))
    masked_img = img_input * (1-mask_input) + mask_input * transform_n(blur_part)
    return masked_img

def score_gen (transform, img_input, mask_input, model):
    masked_img = transform(img_input, mask_input)
    # plot_image(masked_img)

    # put through model
    with torch.no_grad():
        y_output = model(masked_img.unsqueeze(0))

    softmax = nn.Softmax()
    y_softmax = softmax(y_output)


    k = 5

    confidences = np.squeeze(y_output)
    inds = np.argsort(-confidences)
    top_k = inds[:k]
    data_csv = []

    lable_name = pd.read_csv('val_250/lables.txt', sep = ':')
    class_names = []
    for i in range(1000):
        class_names.append(lable_name.iloc[i-1, 1].rstrip(","))

    for i, ind in enumerate(top_k):
        data_csv.append([class_names[ind], 100*y_softmax[0,ind].item()])
        # print(f'Class #{i + 1} - {class_names[ind]} - Logit: {y_output[0,ind]:.2f} - Softmax: {100*y_softmax[0,ind]:.2f}%')
    return data_csv
