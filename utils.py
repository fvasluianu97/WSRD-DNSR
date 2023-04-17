import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
from math import log10
from skimage import color
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.filters import threshold_otsu

from PIL import Image, ImageFilter


class Tools:
    pil = transforms.ToPILImage()
    gray = transforms.Grayscale(num_output_channels=1)


def compute_loader_otsu_mask(img, img_free):
    im_free = Tools.gray(img_free)
    im_ = Tools.gray(img)
    diff = np.abs(np.asarray(im_free, dtype='float32') - np.asarray(im_, dtype='float32'))
    img_diff = Image.fromarray(np.uint8(diff))
    max_diff = img_diff.filter(ImageFilter.MaxFilter(size=3))
    diff = np.asarray(max_diff)
    thresh = threshold_otsu(diff)
    mask = Image.fromarray(np.uint8((diff >= thresh) * 255)) 
    return mask


def compute_otsu_mask(img, img_free):
    im_free = Tools.gray(Tools.pil((img_free.data).cpu()))
    im_ = Tools.gray(Tools.pil((img.data).cpu()))
    diff = np.abs(np.asarray(im_free, dtype='float32') - np.asarray(im_, dtype='float32'))
    img_diff = Image.fromarray(np.uint8(diff))
    max_diff = img_diff.filter(ImageFilter.MaxFilter(size=3))
    diff = np.asarray(max_diff)
    thresh = threshold_otsu(diff)
    mask = torch.tensor(np.float32(diff >= thresh)).unsqueeze(0).cuda()
    mask.requires_grad = False
    return mask


def rescale_lab(lab_img):
    lab_img[:, :, 0] *= 255.0/100.0
    lab_img[:, :, 1] = (lab_img[:, :, 1] + 127.0)
    lab_img[:, :, 2] = (lab_img[:, :, 2] + 127.0)
    return lab_img


def psnr_lab(lab_img_out, lab_img_gt):
    lab_img_gt = rescale_lab(lab_img_gt)
    lab_img_out = rescale_lab(lab_img_out)
    rmse = np.sqrt(mean_squared_error(lab_img_gt, lab_img_out))
    return 20 * log10(255 / rmse)


def rgb2gray(image):
    rgb_image = 255 * image
    return 0.299 * rgb_image[0, :, :] + 0.587 * rgb_image[1, :, :] + 0.114 * rgb_image[2, :, :]


def rgb2lab(numpy_rgb_img):
    numpy_rgb_img = numpy_rgb_img.transpose((2, 1, 0))
    lab_img = color.rgb2lab(numpy_rgb_img)
    return lab_img


def analyze_image_pair(synthetic_image, expected_image):
    synthetic_image = synthetic_image.detach().cpu().data
    expected_image = expected_image.detach().cpu().data

    mse_loss = F.mse_loss(synthetic_image, expected_image)
    rmse_loss = torch.sqrt(mse_loss).item()
    psnr = 20 * log10(1 / rmse_loss)
    return rmse_loss, psnr


def analyze_image_pair_rgb(synthetic_image, expected_image):
    synthetic_image = 255.0 * synthetic_image.detach().cpu().data
    expected_image = 255.0 * expected_image.detach().cpu().data

    mse_loss = F.mse_loss(synthetic_image, expected_image)
    rmse_loss = torch.sqrt(mse_loss).item()
    psnr = 20 * log10(255 / rmse_loss)
    return rmse_loss, psnr


def analyze_image_pair_lab(synthetic_image, expected_image):
    synthetic_image = (255. * synthetic_image.detach().cpu().numpy()).astype(np.uint8)
    expected_image = (255. * expected_image.detach().cpu().numpy()).astype(np.uint8)

    lab_syn_img = rgb2lab(synthetic_image)
    lab_gt_img = rgb2lab(expected_image)

    mse_loss = mean_squared_error(lab_syn_img, lab_gt_img)
    rmse_loss = np.sqrt(mse_loss)

    psnr = psnr_lab(lab_syn_img, lab_gt_img)
    return rmse_loss, psnr


def compute_shadow_mask(shadow_image, shadow_free_image):
    data_size = shadow_image.size()
    batch_size = data_size[0]
    batch_mask = []
    for i in range(batch_size):
        diff = shadow_free_image[i, :, :, :] - shadow_image[i, :, :, :]
        diff = rgb2gray(diff)
        thresh = torch.median(diff)
        diff = ((diff >= thresh).float()).unsqueeze_(0)
        batch_mask.append(diff)
    return torch.stack(batch_mask)


def compute_shadow_mask_symmetric(shadow_image, shadow_free_image):
    data_size = shadow_image.size()
    batch_size = data_size[0]
    batch_mask = []
    for i in range(batch_size):
        """Rescale up to [0, 1]"""
        shadow_img = (0.5 * (1 + shadow_image[i, :, :, :])).cuda()
        shadow_free_img = (0.5 * (1 + shadow_free_image[i, :, :, :])).cuda()
        diff = shadow_free_img - shadow_img
        diff = rgb2gray(diff)
        thresh = torch.median(diff)
        diff = ((diff < thresh).float() * -2.0).unsqueeze_(0)
        support = torch.ones(1, data_size[2], data_size[3]).cuda()
        diff = diff + support
        batch_mask.append(diff)
    return torch.stack(batch_mask).cuda()


def compute_shadow_mask_otsu(shadow_image, shadow_free_image):
    batch_size = shadow_image.size()[0]
    batch_mask = []

    for i in range(batch_size):
        batch_mask.append(compute_otsu_mask(shadow_image[i, :, :, :], shadow_free_image[i, :, :, :]))

    return torch.stack(batch_mask).cuda()
