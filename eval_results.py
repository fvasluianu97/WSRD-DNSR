import os
import numpy as np
from skimage.io import imread
from skimage import color
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

def mresize(image, imsize, anti_aliasing=True):
    imshape = image.shape
    h = imshape[0]
    w = imshape[1]

    if h != imsize[0] or w != imsize[1]:
        image = resize(image, imsize, anti_aliasing=anti_aliasing)
    return image


if __name__ == '__main__':
    inp_dir = "datasets/ISTD/test/test_A"
    mask_dir = "datasets/ISTD/test/test_B"
    res_dir = "results_v2/results/Shformer" # "datasets/ISTD+/test/test_A"
    gt_dir = "datasets/ISTD/test/test_C"
    # imsize = (256, 256)
    imsize = (480, 640)
    num_pixels = imsize[0] * imsize[1]
    use_Lab = True

    file_names = sorted([f for f in os.listdir(res_dir) if f.endswith(".png")])
    num_samples = len(file_names)

    rmse = 0
    shrmse = 0
    frmse = 0
    psnr = 0
    ssim = 0

    for f in file_names:
        inp_img = mresize(imread("{}/{}".format(inp_dir, f)), imsize, anti_aliasing=True)
        out_img = imread("{}/{}".format(res_dir, f))
        gt_img = mresize(imread("{}/{}".format(gt_dir, f)), imsize, anti_aliasing=True)
        mask = mresize(imread("{}/{}".format(mask_dir, f)), imsize, anti_aliasing=True) // 255

        mask3 = np.zeros(out_img.shape)
        for k in range(3):
            mask3[:, :, k] = mask

        psnr += peak_signal_noise_ratio(gt_img, out_img)
        ssim += structural_similarity(gt_img, out_img, channel_axis=-1)

        if use_Lab:
            inp_cmp = color.rgb2lab(inp_img)
            out_cmp = color.rgb2lab(out_img)
            gt_cmp = color.rgb2lab(gt_img)
        else:
            inp_cmp = inp_img
            out_cmp = out_img
            gt_cmp = gt_img

        inv_sh_ratio = num_pixels / np.sum(mask)
        inv_shf_ratio = num_pixels / (num_pixels - np.sum(mask))

        rmse += np.sqrt(mean_squared_error(out_cmp, gt_cmp))
        shrmse += np.sqrt(mean_squared_error(mask3 * out_cmp, mask3 * gt_cmp) * inv_sh_ratio)
        frmse += np.sqrt(mean_squared_error((1 - mask3) * out_cmp, (1 - mask3) * gt_cmp) * inv_shf_ratio)

    psnr /= num_samples
    ssim /= num_samples
    rmse /= num_samples
    shrmse /= num_samples
    frmse /= num_samples
    print(psnr, ssim, rmse, shrmse, frmse)


