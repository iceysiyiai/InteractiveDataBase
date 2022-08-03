import torch.nn as nn
from torchvision import transforms, utils

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

transform_g=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.ToTensor(),
                              AddGaussianNoise(0.15, 0.2)  
                           ])

def transform_Gaussian(img_input, mask_input):
  blur_part = mask_input * img_input
  masked_img = img_input * (1-mask_input) + mask_input * transform_g(blur_part)
  return masked_img

import torchvision.transforms as T
transform_n = T.GaussianBlur(kernel_size=(17, 17), sigma=(0.1, 200))

def transform_Noise(img_input, mask_input):
  blur_part = mask_input * img_input
  masked_img = img_input * (1-mask_input) + mask_input * transform_n(blur_part)
  return masked_img

def score_gen (transform, img_input, mask_input, model):

  masked_img = transform(img_input, mask_input)
  plot_image(masked_img)

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

  print(f'Correct Prediction: {class_names[label]}')
  print('Preservation Model:')
  for i, ind in enumerate(top_k):
    data_csv.append([class_names[ind], 100*y_softmax[0,ind].item()])
    print(f'Class #{i + 1} - {class_names[ind]} - Logit: {y_output[0,ind]:.2f} - Softmax: {100*y_softmax[0,ind]:.2f}%')
  return data_csv