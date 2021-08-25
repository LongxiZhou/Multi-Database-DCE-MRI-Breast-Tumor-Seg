import Tool_Functions.Functions as Functions
import os
import numpy as np


def tumor_center_z(tumor_mask):
    tumor_loc = np.where(tumor_mask > 0.5)
    return int(np.median(tumor_loc[2]))


def get_enhanced(file_name):
    print('processing', file_name)
    a = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/rescaled_array_hayida/' + file_name[:-4] + '.npz')['array']
    p = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/probability_map_stage_one/' + file_name[:-4] + '.npy')

    enhanced = np.zeros([464, 464, 240, 5], 'float32')
    enhanced[:, :, :, 0: 3] = a[:, :, :, 0: 3]  # data channel
    enhanced[:, :, :, 4] = a[:, :, :, 3]  # gt mask

    std = np.std(p)

    roi = np.array(p > 3 * std, 'float32')

    enhanced[:, :, :, 3] = roi  # enhanced_channel

    Functions.save_np_array('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/enhanced_arrays/', file_name[:-4], enhanced,
                            compress=True)


def plot_tumor(file_name):
    print('processing', file_name)
    a = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/rescaled_array_hayida/' + file_name)['array']
    p = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/probability_map_stage_one/' + file_name[:-4] + '.npy')

    std = np.std(p)

    loc = tumor_center_z(a[:, :, :, 3])

    gt_mask = np.array(a[:, :, loc, 3] > 0.5, 'float32')

    roi = np.array(p > 3 * std, 'float32')[:, :, loc]

    t_p = roi * gt_mask

    f_p = np.array(roi - gt_mask > 0.5, 'float32')

    f_n = np.array(gt_mask - roi > 0.5, 'float32')

    image = a[:, :, loc, 0]

    image = Functions.cast_to_0_1(image)

    output = np.zeros([464, 464 * 2, 3], 'float32')

    output[:, 0: 464, 0] = image
    output[:, 0: 464, 1] = image
    output[:, 0: 464, 2] = image

    output[:, 464::, :] = output[:, 0: 464, :]

    output[:, 464::, 0] -= t_p
    output[:, 464::, 1] += t_p
    output[:, 464::, 2] -= t_p  # G

    output[:, 464::, 0] += f_n
    output[:, 464::, 1] -= f_n
    output[:, 464::, 2] -= f_n  # R

    output[:, 464::, 0] -= f_p
    output[:, 464::, 1] -= f_p
    output[:, 464::, 2] += f_p  # R

    output[:, 0: 464, :] = np.swapaxes(output[:, 0: 464, :], 0, 1)
    output[:, 464::, :] = np.swapaxes(output[:, 464::, :], 0, 1)

    output = np.clip(output, 0, 1)
    Functions.image_save(output, '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/stage_one_harbin_std3/' +
                         file_name[:-4], high_resolution=True)
    return None


def analysis_prediction(rescaled_array, predict_mask, ground_truth, image_save_path=None):
    a = rescaled_array
    p = predict_mask

    loc = tumor_center_z(ground_truth)

    gt_mask = ground_truth[:, :, loc]

    p = p[:, :, loc]

    t_p = p * gt_mask

    f_p = np.array(p - gt_mask > 0.5, 'float32')

    f_n = np.array(gt_mask - p > 0.5, 'float32')

    image = a[:, :, loc, 0]

    image = Functions.cast_to_0_1(image)

    output = np.zeros([464, 464 * 2, 3], 'float32')

    output[:, 0: 464, 0] = image
    output[:, 0: 464, 1] = image
    output[:, 0: 464, 2] = image

    output[:, 464::, :] = output[:, 0: 464, :]

    output[:, 464::, 0] -= t_p
    output[:, 464::, 1] += t_p
    output[:, 464::, 2] -= t_p  # G

    output[:, 464::, 0] += f_n
    output[:, 464::, 1] -= f_n
    output[:, 464::, 2] -= f_n  # R

    output[:, 464::, 0] -= f_p
    output[:, 464::, 1] -= f_p
    output[:, 464::, 2] += f_p  # R

    output[:, 0: 464, :] = np.swapaxes(output[:, 0: 464, :], 0, 1)
    output[:, 464::, :] = np.swapaxes(output[:, 464::, :], 0, 1)

    output = np.clip(output, 0, 1)
    if image_save_path is None:
        Functions.image_show(output)
    else:
        Functions.image_save(output, image_save_path, high_resolution=True)
    return None


def size_stat(file_name):
    print('processing', file_name)
    a = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/rescaled_array_hayida/' + file_name)['array']

    gt = a[:, :, :, 3]

    volume = np.sum(gt) * 0.446 * 0.446 * 1 / 10 / 10 / 10

    print(volume)
    return volume


def min_std(total, mean, fail):
    l = []
    p = total * mean - fail * 0.2
    for i in range(fail):
        l.append(0.2)
    for i in range(total - fail):
        l.append(p / (total - fail))
    print(np.std(l), np.average(l))


def max_std(total, mean, fail):
    l = []
    p = total * mean
    for i in range(fail):
        l.append(0)
    y = p - 0.2 * (total - fail)
    y = round(y / 0.8)
    for i in range(y):
        l.append(1)
    for i in range(total - y - fail):
        l.append(0.2)
    print(np.std(l), np.average(l), np.sum(np.array(l) < 0.2), len(l))
    min_std(total, mean, fail)


if __name__ == "__main__":
    fn_list = os.listdir('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/visualization/stage_one_harbin_std3/')

    l = []
    for fn in fn_list:
        get_enhanced(fn)

    exit()

    print(np.median(l), np.average(l), np.std(l), np.min(l), np.max(l))
    l.sort()
    print(l)
    l.reverse()
    print(l)
