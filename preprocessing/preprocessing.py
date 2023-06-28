import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn.image import resample_img
import matplotlib.pyplot as plt
import h5py
from joblib import Parallel, delayed
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.filters import sobel
from skimage.morphology import remove_small_objects
import random
from math import floor
from scipy.ndimage import rotate, zoom
from skimage.transform import resize
import cv2
from glob import glob
import argparse
from functools import partial
from sys import exc_info
import os


def segment(arr, thresh=-600, background=-1024):
    """Segment examination table from image. Threshold was chosen manually"""
    # make copies
    thresh_arr = np.copy(arr)
    res_arr = np.copy(arr)

    # threshold arrays into foreground and background. Value is
    res_arr[arr < thresh] = background
    thresh_arr[arr < thresh] = 1
    thresh_arr[arr >= thresh] = 2

    # Edge detection, watershed-segmentation, fill holes
    elevation_map = sobel(arr)
    segmentation = watershed(elevation_map, thresh_arr)
    fill = binary_fill_holes(segmentation - 1)

    # label segmented regions. -1024 is background (label 0).
    labels = label(fill, background=background)

    # collect properties of segmented regions. Find largest object that is not background. Fill holes again.
    rp = regionprops(labels)
    size = max([i.area for i in rp])
    labels = remove_small_objects(labels, min_size=size - 1)
    labels = binary_fill_holes(labels)

    # keep array values where it is labeled, set the rest to air
    res_arr = np.where(labels, res_arr, background)
    return res_arr


def normalize(arr, vmin=-1024, vmax=1024):
    """Normalize an array between 0 and 1"""
    # normalize arr
    arr = np.clip(arr, vmin, vmax)

    arr -= vmin
    arr *= (1 / (vmax - vmin))
    arr = np.clip(arr, 0, 1)
    return arr


def augmentation(arr, percentage=0.5, left_shift=-1, down_shift=-1, background=0,
                 randresize=False, randrotate=0, noise=False):
    """Augment an array by cropping random patches of edge size size(array)*percentage.
       Randomly resize and rotate or add gaussian noise"""
    w, h = arr.shape

    if randrotate != 0:
        angle = random.randint(-randrotate, randrotate)
        arr = rotate(arr, angle)
        arr = resize(arr, (w, h))

    if randresize:
        resize_factor = 1 + 0.5*random.random()
    else:
        resize_factor = 1

    w_new = int(floor(w * percentage))
    h_new = int(floor(h * percentage))

    if left_shift:
        left_shift = random.randint(0, int((w - w_new)))
    if down_shift:
        down_shift = random.randint(0, int((h - h_new)))

    patch = np.array(arr[left_shift:(w_new + left_shift), down_shift:(h_new + down_shift)])
    patch = np.array(zoom(patch, resize_factor, order=0))

    w_new, h_new = patch.shape
    left_shift = random.randint(0, int((w - w_new)))
    down_shift = random.randint(0, int((h - h_new)))

    crop_arr = np.full((w, h), background)
    crop_arr[left_shift:(w_new + left_shift), down_shift:(h_new + down_shift)] = patch

    if noise:
        row, col = crop_arr.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        crop_arr = cv2.add(crop_arr, gauss)

    return crop_arr


def get_dirs(work_dir, df, min_age, max_age):
    """Get all directories containing image data. Has to be rewritten according to data structure."""
    dirs_to_draw = []
    female = []
    male = []
    for study_dir in work_dir.iterdir():
        if study_dir.is_dir():
            pat_hash = study_dir.name
            try:
                loc = df[df['Subject ID'] == pat_hash].index[0].item()
                age = str(df['age'][loc])
                age = int(age.split("Y")[0])
                if min_age <= int(age) <= max_age and df['diagnosis'][loc] == "NEGATIVE":
                    if df['sex'][loc] == "F":
                        female.append(age)
                    if df['sex'][loc] == "M":
                        male.append(age)
                    dirs_to_draw.append(str(study_dir))
            except Exception as e:
                print(e)
        else:
            continue

    print("total: " + str(len(dirs_to_draw)) + "\n" + "first: " + dirs_to_draw[0] + "\n" + "last: " +
          dirs_to_draw[-1])

    male = np.array(male)
    female = np.array(female)
    print("Number of male subjects: " + str(len(male)) + "\nMean age: "
          + str(np.mean(male)) + "+/-" + str(np.std(male)))
    print("Number of female subjects: " + str(len(female)) + "\nMean age: "
          + str(np.mean(female)) + "+/-" + str(np.std(female)))

    return dirs_to_draw


def get_file(dir, df):
    """Load image file from directory. Has to be rewritten according to project structure and database."""
    study_dir = Path(dir)
    pat_hash = study_dir.name
    file_name = glob(str(study_dir) + "/**/CT.nii.gz", recursive=True)[0]

    ret_loc = df[df['Subject ID'] == pat_hash].index[0].item()
    ret_age = int(df['age'][ret_loc].split("Y")[0])
    img = nib.load(file_name)

    return pat_hash, file_name, ret_age, img


def chunky(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        if not (i + n) > len(lst):
            so_chunky = lst[i:i + n]
        else:
            so_chunky = lst[i:len(lst)]
        yield so_chunky


def visualize(comb_arr, pat_hash, age):
    # save visualization
    img_dir = Path("./images")
    fig, ax = plt.subplots()
    fig.suptitle(pat_hash)
    ax.set_title("MIP " + str(age) + "y")
    ax.imshow(np.rot90(comb_arr[0, 0, :, :]), cmap="bone", vmin=0, vmax=1)
    out_file_name = pat_hash + ".png"
    out_file = img_dir / out_file_name
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def proc(df, thickness, augment, channels, shape, tpl):
    """Preprocess one study"""
    # print(df, thickness, augment, channels, shape, tpl)
    i = tpl[0]
    ct_min = -1024

    try:
        pat_hash, file_name, ret_age, img = get_file(tpl[1], df)

        # read out original spacing and shape
        orig_spacing = np.array(img.header.get_zooms())
        orig_shape = img.header.get_data_shape()

        # specify required shape for network and compute spacing needed
        target_shape = (112, 112, 224)
        target_spacing = (orig_spacing[0] * (orig_shape[0] / target_shape[0]),
                          orig_spacing[1] * (orig_shape[1] / target_shape[1]),
                          orig_spacing[2] * (orig_shape[2] / target_shape[2]))
        # change image affine
        target_affine = np.copy(img.affine)
        target_affine[:3, :3] = np.diag(target_spacing / orig_spacing) @ img.affine[:3, :3]

        # resample image and get array data
        res_img = resample_img(img,
                               target_affine=target_affine,
                               target_shape=target_shape,
                               interpolation="continuous",
                               fill_value=ct_min)
        res_arr = np.array(res_img.get_fdata())

        # segmentation
        res_arr = segment(res_arr)
        # compute maximum intensity projection
        mip_sagittal = np.max(res_arr[(res_arr.shape[0] // 2 - thickness//2):(res_arr.shape[0] // 2 + thickness//2), :, :],
                              axis=0)
        mip_coronar = np.max(res_arr[:, (res_arr.shape[1] // 2 - thickness//2):(res_arr.shape[1] // 2 + thickness//2), :],
                             axis=1)
        mip_combined = np.concatenate((mip_coronar, mip_sagittal), axis=0)
        mip_combined = normalize(mip_combined, vmin=max(ct_min, np.min(mip_combined)), vmax=np.max(mip_combined))

        comb_arr = np.zeros((augment, channels, shape, shape))
        comb_arr[0, 0, :, :] = mip_combined

        print(i, pat_hash, comb_arr.shape, ret_age)

        return i, comb_arr, pat_hash, ret_age

    except Exception as e:
        exc_type, exc_obj, exc_tb = exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return i, np.zeros((augment, channels, shape, shape)), 99999999999, 9999


def run_preprocessing(work_dir, info_file, h5_file, min_age, max_age,
                      thickness, jobs, augment, channels, shape, save):
    """Run the preprocessing"""
    work_dir = Path(work_dir)
    df = pd.read_csv(info_file)

    dirs_to_draw = get_dirs(work_dir=work_dir, df=df, min_age=min_age, max_age=max_age)[:10]

    dir_nums = np.arange(0, len(dirs_to_draw))
    i_dirs = list(zip(dir_nums, dirs_to_draw))

    result_arr = np.zeros((len(dirs_to_draw) * augment, channels, shape, shape), dtype=np.float32)
    info_dict = {"layer": [], "pat_hash": [], "age": []}
    error_dict = {"layer": [], "pat_hash": [], "cause": []}

    # break down patients ("k") into chunks for multiprocessing
    k_chunks = list(chunky(i_dirs, jobs))

    chunk_i = 0
    num_chunks = len(dirs_to_draw) // jobs

    if save:
        img_dir = Path("./images/")
        img_dir.mkdir(exist_ok=True)

    for chunk in k_chunks:
        chunk_i = chunk_i + 1
        print("chunk " + str(chunk_i) + " of " + str(num_chunks + 1))
        results = Parallel(n_jobs=min(jobs, len(chunk)))(delayed(partial(proc, df, thickness, augment, channels, shape))
                                                                 (tpl) for tpl in chunk)
        for result in results:
            if not result[2] == 9999999999:
                result_arr[result[0]:result[0]+augment, :, :, :] = result[1]
                info_dict["layer"].append(result[0] * augment)
                info_dict["pat_hash"].append(result[2])
                info_dict["age"].append(result[3])

                if save:
                    visualize(result[1], result[2], result[3])

            else:
                error_dict["layer"].append(result[0])
                error_dict["pat_hash"].append(result[2])
                error_dict["cause"].append(result[3])

    with h5py.File(h5_file, "a") as hf:
        dataset = hf.require_dataset(name="dataset",
                                     data=result_arr,
                                     shape=result_arr.shape,
                                     dtype=result_arr.dtype)

    out_df = pd.DataFrame.from_dict(info_dict)
    out_df.to_csv("info.csv")
    print(error_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wor", help="Path to working directory", required=True)
    parser.add_argument("--inf", help="File with patient information", required=True)
    parser.add_argument("--h5f", help="destination h5-file", required=True)
    parser.add_argument("--min", help="minimum age to include", default=20, type=int)
    parser.add_argument("--max", help="maximum age to include", default=85, type=int)
    parser.add_argument("--jbs", help="parallel jobs", default=10, type=int)
    parser.add_argument("--thk", help="number of layers for MIP-computation", default=10, type=int)
    parser.add_argument("--aug", help="number of augmentations per original image", default=0, type=int)
    parser.add_argument("--shp", help="output shape of 2D inputs", default=224, type=int)
    parser.add_argument("--chn", help="number of channels to use", default=1, type=int)
    parser.add_argument("--sav", help="save images of inputs", default=False)

    arguments = parser.parse_args()

    run_preprocessing(work_dir=arguments.wor,
                      info_file=arguments.inf,
                      h5_file=arguments.h5f,
                      min_age=arguments.min,
                      max_age=arguments.max,
                      jobs=arguments.jbs,
                      thickness=arguments.thk,
                      augment=arguments.aug,
                      channels=arguments.chn,
                      shape=arguments.shp,
                      save=arguments.sav)


if __name__ == '__main__':
    main()
