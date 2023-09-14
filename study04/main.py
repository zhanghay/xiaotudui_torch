#!/usr/bin/env python
# @Time    : 2021/3/17 14:55
# @File    : main.py
# @Software: PyCharm
# @Language: Python
# @Version : 1.0
# @Usage   : transform image to tensor, normalize image, reszie image
#            save image to tensorboard
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "../dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
totensor = transforms.ToTensor()
tensor = totensor(img)
writer = SummaryWriter("mylog")
writer.add_image("ants", tensor)

# 归一化
tran_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 3 个 channel
# ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
tensor_normalized = tran_norm(tensor)
print(tensor[0][0][0])
print(tensor_normalized[0][0][0])
writer.add_image("ants_normal", tensor_normalized)

# 调整大小
tran_resize = transforms.Resize((224, 224))
tensor_resized = tran_resize(tensor)
writer.add_image("ants_resize", tensor_resized)
# writer.close()

# 调整大小2
tran_resize2 = transforms.Resize((224, 224))
tensor_resized2 = tran_resize2(img)
tensor_resized2 = totensor(tensor_resized2)
writer.add_image("ants_resize2", tensor_resized2)
# writer.close()

# Compose
tran_resize3 = transforms.Resize(64)
tran_compose = transforms.Compose([tran_resize3, totensor])
# Composes several transforms together
tensor_compose = tran_compose(img)
writer.add_image("ants_compose", tensor_compose)
writer.close()
