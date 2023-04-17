import os
import numpy as np
from skimage.io import imread
from skimage import color
from skimage.metrics import mean_squared_error
from skimage.transform import resize


if __name__ == '__main__':
    inp_dir = "datasets/ISTD+/test/test_A"
    mask_dir = "datasets/ISTD+/test/test_B"
    res_dir = "results/AEF+"
    gt_dir = "datasets/ISTD+/test/test_C"
    imsize = (256, 256)
    # imsize = (480, 640)

    file_names = sorted([f for f in os.listdir(res_dir) if f.endswith(".png")])
    num_samples = len(file_names)

    rmse = 0
    shrmse = 0
    frmse = 0

    for f in file_names:
        inp_img = resize(imread("{}/{}".format(inp_dir, f)), imsize, anti_aliasing=True)
        out_img = imread("{}/{}".format(res_dir, f))
        gt_img = resize(imread("{}/{}".format(gt_dir, f)), imsize, anti_aliasing=True)
        mask = resize(imread("{}/{}".format(mask_dir, f)), imsize, anti_aliasing=True)

        mask3 = np.zeros(out_img.shape)
        for k in range(3):
            mask3[:, :, k] = mask

        inp_lab = color.rgb2lab(inp_img)
        out_lab = color.rgb2lab(out_img)
        gt_lab = color.rgb2lab(gt_img)

        rmse += np.sqrt(mean_squared_error(out_lab, gt_lab))
        shrmse += np.sqrt(mean_squared_error(mask3 * out_lab, mask3 * gt_lab))
        frmse += np.sqrt(mean_squared_error((1 - mask3) * out_lab, (1 - mask3) * gt_lab))

    rmse /= num_samples
    shrmse /= num_samples
    frmse /= num_samples
    print(rmse, shrmse, frmse)


