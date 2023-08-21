import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import random
import os

# "key words: primeval forest, beach", "garden", "suburb"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


def setup_seed(seed):
     # 设置随机数种子
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


prompt_list = ['garden']
num_of_imgs_for_each_class = 10
img_save_dir = 'data/cover' # the dir for saving the generated cover images


for prompt in prompt_list:
     img_dir = os.path.join(img_save_dir, prompt)
     if not os.path.exists(img_dir):
          os.makedirs(img_dir)

for prompt in prompt_list:
     for i in range(num_of_imgs_for_each_class):
          random_sedd = i + 1   
          setup_seed(random_sedd)
          image = pipe(prompt).images[0]
          
          img_name = str(i) + '.png'
          img_dir = os.path.join(img_save_dir, prompt)
          image.save(os.path.join(img_dir, img_name))