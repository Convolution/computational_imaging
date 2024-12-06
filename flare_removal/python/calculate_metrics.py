# calculate_metric.py
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import math
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('gt_dir', None,
                    'The directory contains all ground truth test images.')
flags.DEFINE_string('blended_dir', None,
                    'The directory contains all blended images.')
flags.DEFINE_string('out_dir', None, 'Output directory.')


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # Images are identical
        return float('inf')
    pixel_max = 255.0
    psnr = 20 * math.log10(pixel_max / math.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim, _ = compare_ssim(img1_gray, img2_gray, full=True)
    return ssim

def main(_):
    # Directories
    gt_dir = FLAGS.gt_dir  # Change this to your reference image directory if needed
    blended_dir = FLAGS.blended_dir  # Directories to compare
    out_dir = FLAGS.out_dir
    results = []

    #for out_dir in blended_dir:
    #results[out_dir] = []
    #output_path = blended_dir
    for img_name in os.listdir(blended_dir):
        # Load reference and corresponding output images
        input_img_path = os.path.join(gt_dir, img_name)
        output_img_path = os.path.join(blended_dir, img_name)

        if not os.path.exists(input_img_path) or not os.path.exists(output_img_path):
            continue  # Skip if files don't match

        input_img = cv2.imread(input_img_path)
        output_img = cv2.imread(output_img_path)

        if input_img is None or output_img is None:
            continue  # Skip invalid images

        # Compute PSNR and SSIM
        psnr = calculate_psnr(input_img, output_img)
        ssim = calculate_ssim(input_img, output_img)
        results.append((img_name, psnr, ssim))

    output_file = os.path.join(out_dir, "psnr_ssim_results.txt")
    with open(output_file, "w") as file:
        i = 0
        avg_psnr = 0
        avg_ssim = 0
        file.write(f"Results for {out_dir}:\n")
        for img_name, psnr, ssim in results:
            file.write(f"Image: {img_name} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}\n")
            i += 1
            avg_psnr += psnr
            avg_ssim += ssim

        avg_psnr /= i
        avg_ssim /= i
        file.write(f"Average PSNR: {avg_psnr:.2f} | Average SSIM: {avg_ssim:.4f}\n")

print("Results have been saved to their respective model testing folders.")

if __name__ == '__main__':
  app.run(main)
