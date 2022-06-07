import numpy as np


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def gaussian_kernel(std_dev=1):
    row_size = std_dev * 3 * 2 + 1
    x = np.linspace(-row_size / 3, row_size / 3, row_size)
    gaus = np.exp(-((x / std_dev) ** 2))
    norm_gaus = gaus / sum(gaus)
    kernel = norm_gaus[:, np.newaxis] * norm_gaus[np.newaxis, :]
    return kernel


def out_of_focus_kernel(r=9):
    shape = (2 * r + 1, 2 * r + 1)
    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    kernel = np.zeros(shape)

    cx = shape[0] // 2
    cy = shape[1] // 2
    mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 <= r**2
    kernel[mask] = 1 / (np.pi * r**2)
    return kernel


def horizontal_motion_blur(l=9):
    return (np.ones(l) * 1 / l)[:, np.newaxis]


def add_gaussian_noise(arr, percent_noise=0.01):
    error = np.random.normal(loc=0, scale=1, size=arr.shape)
    error = error / np.linalg.norm(error)
    return arr + percent_noise * np.linalg.norm(arr) * error


def salt_and_pepper_noise(arr, percent):
    # assume its between 0 and 1 the values
    assert 0 < percent < 1.0
    size = arr.size
    portion = int(size * percent)
    noise_coord = np.random.choice(size, size=portion, replace=False)
    arr[noise_coord[portion // 2 :]] = 1
    arr[noise_coord[: portion // 2]] = 0
    return arr
