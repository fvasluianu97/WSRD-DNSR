import argparse
import numpy as np
import os
import torch
from dconv_model import DistillNet
from initializer import weights_init_normal
from ImageLoaders import PairedImageSet
from skimage.metrics import structural_similarity as ssim
from torch.autograd import Variable
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from UNet import UNetTranslator
from utils import analyze_image_pair, analyze_image_pair_rgb, analyze_image_pair_lab, compute_shadow_mask,\
    compute_shadow_mask_otsu

def torch2numpy(img):
    return 255. * torch.clamp(img.squeeze(0).cpu().detach(), min=0, max=1).numpy().transpose((2, 1, 0)).astype(np.uint8)


if __name__ == '__main__':
    # parse CLI arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=int, default=2, help="[0]UNet [else]DistillNet")
    parser.add_argument("--fullres", type=int, default=1, help="[0]inference with maxres [1]fullres inference")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--model_dir", default="checkpoints/DNSR-fullres-trlearning-rcrop-r9",
                        help="Dir for model evaluation.")
    parser.add_argument("--image_dir", default="results/DNSR-wsrd2-r4",
                        help="Path for the directory used to save the output test images")
    opt = parser.parse_args()
    print(opt)

    print('CUDA: ', torch.cuda.is_available(), torch.cuda.device_count())


    criterion_pixelwise = torch.nn.MSELoss()

    if opt.model_type == 0:
        translator = UNetTranslator(in_channels=3, out_channels=3)
        translator.apply(weights_init_normal)
    else:
        translator = torch.nn.DataParallel(DistillNet(num_iblocks=6, num_ops=4))
        translator.load_state_dict(torch.load("{}/gen_sh2f.pth".format(opt.model_dir)))

    cuda = torch.cuda.is_available()
    if cuda:
        print("USING CUDA FOR MODEL EVALUATION")
        translator.cuda()
        criterion_pixelwise.cuda()



    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    test_il = PairedImageSet('../NTIRE-SR/data/WSRD2', 'test', size=None, use_mask=False, aug=False)

    val_dataloader = DataLoader(
        test_il,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu
    )

    val_samples = len(val_dataloader)

    srmse = 0
    spsnr = 0

    with torch.no_grad():
        translator = translator.eval()

        os.makedirs("{}".format(opt.image_dir), exist_ok=True)

        for idx, (B_img, AB_mask, A_img) in enumerate(val_dataloader):
            inp = Variable(A_img.type(Tensor))
            gt = Variable(B_img.type(Tensor))
            mask = Variable(AB_mask.type(Tensor))

            if opt.fullres == 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    target_size = (960, 1280)
                    inp = interpolate(inp, target_size, mode='bicubic')
                    gt = interpolate(gt, target_size, mode='bicubic')
                    mask = interpolate(mask, target_size, mode='nearest')
                    out = translator(inp, mask)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    b, c, h, w = inp.shape
                    target_size = (960, 1280)
                    res_inp = interpolate(inp, target_size, mode='bicubic')
                    res_mask = interpolate(mask, target_size, mode='nearest')

                    dsz_out = translator(res_inp, res_mask)
                    out = interpolate(dsz_out, (h, w), mode='bicubic')

            rmse, psnr = analyze_image_pair_rgb(out.squeeze(0), gt.squeeze(0))

            srmse += rmse
            spsnr += psnr

            img_synth = torch.clamp(out.detach().data, min=0, max=1)
            img_real = inp.detach().data
            img_gt = gt.detach().data
            img_sample = torch.cat((img_real, img_synth, img_gt), dim=-1)
            save_image(img_sample, "{}/{}_im.png".format(opt.image_dir, idx + 1))
            mask_sample = torch.cat((mask, compute_shadow_mask_otsu(inp, out)), dim=-1)
            save_image(mask_sample, "{}/{}_mask.png".format(opt.image_dir, idx + 1))

        srmse /= len(test_il)
        spsnr /= len(test_il)

        print(srmse, spsnr)
