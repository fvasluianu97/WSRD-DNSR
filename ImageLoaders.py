import os
import numpy as np
import random
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode
from utils import compute_loader_otsu_mask


class ImageSet(data.Dataset):
    def __init__(self, set_path, set_type, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        clean_path_dir = '{}/{}/{}_C'.format(set_path, set_type, set_type)

        self.clean_images_path = []
        self.smats_path = []
        self.original_images_path = []
        self.num_samples = 0

        for dirpath, dnames, fnames \
                in os.walk("{}/{}/{}_A/".format(set_path, set_type, set_type)):
            for f in fnames:
                if f.endswith(".png"):
                    orig_path = os.path.join(dirpath, f)
                    clean_path = os.path.join(clean_path_dir, f)

                    self.clean_images_path.append(clean_path)
                    self.original_images_path.append(orig_path)
                    self.num_samples += 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        shadow_data = Image.open(self.original_images_path[index])
        clean_data = Image.open(self.clean_images_path[index])
        
        return self.transform(clean_data), self.transform(shadow_data)


class PairedImageSet(data.Dataset):
    def __init__(self, set_path, set_type, size=(256, 256), use_mask=True, aug=False):
        self.augment = aug

        self.size = size
        self.use_mask = use_mask

        self.to_tensor = transforms.ToTensor()
        if size is not None:
            self.resize = transforms.Resize(self.size, interpolation=InterpolationMode.BICUBIC)
        else:
            self.resize = None
            
        if use_mask:
            smat_path_dir = '{}/{}/{}_B'.format(set_path, set_type, set_type)
        
        clean_path_dir = '{}/{}/{}_C'.format(set_path, set_type, set_type)

        self.gt_images_path = []
        self.masks_path = []
        self.inp_images_path = []
        self.num_samples = 0

        for dirpath, dnames, fnames \
                in os.walk("{}/{}/{}_A/".format(set_path, set_type, set_type)):
            for f in fnames:
                if f.endswith(".zip"):
                    continue
                orig_path = os.path.join(dirpath, f)
                
                if use_mask:
                    smat_path = os.path.join(smat_path_dir, f)
                    self.masks_path.append(smat_path)

                clean_path = os.path.join(clean_path_dir, f)
                self.gt_images_path.append(clean_path)
                
                self.inp_images_path.append(orig_path)
                self.num_samples += 1

    def __len__(self):
        return self.num_samples

    def augs(self, gt, mask, inp):
        w, h = gt.size
        tl = np.random.randint(0, h - self.size[0])
        tt = np.random.randint(0, w - self.size[1])
        
        gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
        mask = torchvision.transforms.functional.crop(mask, tt, tl, self.size[0], self.size[1])
        inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])

        if random.random() < 0.5:
            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)
            mask = torchvision.transforms.functional.hflip(mask)
        if random.random() < 0.5:
            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)
            mask = torchvision.transforms.functional.vflip(mask)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)
            mask = torchvision.transforms.functional.rotate(mask, angle)

        return gt, mask, inp

    def __getitem__(self, index):
        inp_data = Image.open(self.inp_images_path[index])
        gt_data = Image.open(self.gt_images_path[index])

        if self.use_mask:
            smat_data = Image.open(self.masks_path[index])
        else:
            smat_data = compute_loader_otsu_mask(inp_data, gt_data)

        if self.augment:
            gt_data, smat_data, inp_data = self.augs(gt_data, smat_data, inp_data)
        else:
            if self.resize is not None:
                gt_data = self.resize(gt_data)
                smat_data = self.resize(smat_data)
                inp_data = self.resize(inp_data)

        tensor_gt = self.to_tensor(gt_data)
        tensor_msk = self.to_tensor(smat_data)
        tensor_inp = self.to_tensor(inp_data)

        return tensor_gt, tensor_msk, tensor_inp





