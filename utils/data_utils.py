import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np
from tqdm import tqdm

# class CameraDataset(Dataset):
    
#     def __init__(self, viewpoint_stack, white_background):
#         self.viewpoint_stack = viewpoint_stack
#         for viewpoint_data in self.viewpoint_stack:
#             viewpoint_data.cuda_nocopy()
#         # print(self.viewpoint_stack[0].image.device)
#         # print(self.viewpoint_stack[199].image.device)
#         # print(len(viewpoint_stack))
#         self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
#     def __getitem__(self, index):
#         viewpoint_cam = self.viewpoint_stack[index]
#         if viewpoint_cam.meta_only:
#             with Image.open(viewpoint_cam.image_path) as image_load:
#                 im_data = np.array(image_load.convert("RGBA"))
#             norm_data = im_data / 255.0
#             arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
#             image_load = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
#             resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
#             viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
#             if resized_image_rgb.shape[1] == 4:
#                 gt_alpha_mask = resized_image_rgb[3:4, ...]
#                 viewpoint_image *= gt_alpha_mask
#             else:
#                 viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
#         else:
#             viewpoint_image = viewpoint_cam.image

#         # print(viewpoint_image.device)
#         # print(viewpoint_cam.dtype)

#         return viewpoint_image, viewpoint_cam
    
#     def __len__(self):
#         return len(self.viewpoint_stack)
    
class CameraDataset():
    
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
        for index in tqdm(range(self.__len__())):
            viewpoint_cam = self.viewpoint_stack[index]
            if viewpoint_cam.meta_only:
                with Image.open(viewpoint_cam.image_path) as image_load:
                    im_data = np.array(image_load.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
                image_load = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
                viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
                if resized_image_rgb.shape[1] == 4:
                    gt_alpha_mask = resized_image_rgb[3:4, ...]
                    viewpoint_image *= gt_alpha_mask
                else:
                    viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
                viewpoint_cam.image = viewpoint_image
            # else:
            #     viewpoint_image = viewpoint_cam.image
            viewpoint_cam.cuda_nocopy()
        
    def debuginfo(self):
        print(self.viewpoint_stack[0].image.device)

    def load_batch_indices(self, batch_size):
        indices = torch.randint(0, self.__len__(), (batch_size,))
        return indices

    def __len__(self):
        return len(self.viewpoint_stack)