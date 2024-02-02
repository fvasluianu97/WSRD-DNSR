import lpips
import os
import numpy as np
from skimage.io import imread
from skimage import color
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.metrics import structural_similarity as skssim

def compute_mssim(ref_img, res_img):
    channels = []

    for i in range(3):
        channels.append(skssim(ref_img[:, :, i], res_img[:, :, i], gaussian_weights=True, use_sample_covariance=False))

    return np.mean(channels)


if __name__ == '__main__':
    res_dir = "results_drive/DNSR_ISTD+_RES"

    file_names = sorted([f for f in os.listdir(res_dir) if f.endswith("_im.png")])
    num_samples = len(file_names)
    print(f"Found {num_samples} files")

    mse = 0
    psnr = 0
    ssim = 0
    slpips = 0

    lab_rmse = 0
    lab_shrmse = 0
    lab_frmse = 0
    loss_fn_alex = lpips.LPIPS(net='alex')

    for f in file_names:
        gallery_img = imread(f"{res_dir}/{f}")
        h, w, c = gallery_img.shape
        inp_img = gallery_img[:, :w//3, :]
        out_img = gallery_img[:, w//3: 2*(w//3), :]
        gt_img = gallery_img[:, 2*(w//3):, :]
        mask = 1 / 255.0 * imread(f"{res_dir}/{f.split('_')[0]}_mask.png")[:, :w // 3, :]
        dist01 = loss_fn_alex.forward(lpips.im2tensor(out_img), lpips.im2tensor(gt_img))
        slpips += dist01.item()

        print(f, peak_signal_noise_ratio(out_img, gt_img), dist01)
        mse += mean_squared_error(out_img, gt_img)
        psnr += peak_signal_noise_ratio(out_img, gt_img)
        ssim += compute_mssim(gt_img, out_img)

        ratio1 = h * w / np.sum(mask)
        ratio2 = h * w / np.sum(1 - mask)

        inp_lab = color.rgb2lab(inp_img)
        out_lab = color.rgb2lab(out_img)
        gt_lab = color.rgb2lab(gt_img)

        # lab_rmse += np.sqrt(mean_squared_error(out_lab, gt_lab))
        # lab_shrmse += np.sqrt(mean_squared_error(mask * out_lab, mask * gt_lab) * ratio1)
        # lab_frmse += np.sqrt(mean_squared_error((1 - mask) * out_lab, (1 - mask) * gt_lab) * ratio2)
        h, w, _ = gt_lab.shape
        lab_rmse += 1 / (h * w) * np.sum(np.abs(out_lab - gt_lab))
        lab_shrmse += ratio1 * 1 / (h * w) * np.sum(np.abs(mask * out_lab - mask * gt_lab))
        lab_frmse += ratio2 * 1 / (h * w) * np.sum(np.abs((1 - mask) * out_lab - (1 - mask) * gt_lab))


    mse /= num_samples
    psnr /= num_samples
    ssim /= num_samples

    lab_rmse /= num_samples
    lab_shrmse /= num_samples
    lab_frmse /= num_samples
    slpips /= num_samples

    print(mse, psnr, ssim, slpips)
    print(lab_rmse, lab_shrmse, lab_frmse)

