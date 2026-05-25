import os
import sys
import cv2
import numpy as np
import random

src_path = os.path.abspath(os.path.join("..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from preprocess import preprocess_image_array

CORRUPTIONS = ["noise", "blur", "occlusion"]

def apply_degradation(image: np.ndarray, corruption_type = None, severity: int = 2) -> np.ndarray:
    '''
    Take a corruption and severity to apply to an image array
    If none provided it chooses a random corruption, and a severity of 2. Severity is an integer from 1 to 5 inclusive.
    Return degraded image array
    '''
    # choose random corruption
    if corruption_type is None:
        corruption_type = random.choice(CORRUPTIONS)

    image = image.copy()

    if corruption_type == "noise":
        sigma_map = {
            1: 0.03,
            2: 0.06,
            3: 0.10,
            4: 0.15,
            5: 0.20
        }

        sigma = sigma_map[severity]
        img_float = image.astype(np.float32) / 255.0
        noise = np.random.normal(0, sigma,img_float.shape)
        # add noise
        noisy = img_float + noise
        noisy = np.clip(noisy, 0, 1) # ensure doesnt go outside 0 or 1
        return (noisy * 255).astype(np.uint8)

    elif corruption_type == "blur":
        # code adapted from https://stackoverflow.com/a/57629531
        kernel_size = [3, 5, 7, 9, 11][severity - 1] # kernel size must be odd
        angle = random.randint(0, 180)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1
        matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)

        kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
        kernel /= np.sum(kernel)
        return cv2.filter2D(image, -1, kernel)


    elif corruption_type == "occlusion":
        # failed scan w possible shadow over it (?)
        # apply dark band
        h, w = image.shape
        occluded = image.copy()

        band_width = random.randint(int(w * 0.05), int(w * 0.20))

        x0 = random.randint(0, w - band_width)
        darkness = random.randint(0, 40) # make shadow darkness random
        occluded[:, x0:x0 + band_width] = darkness

        # soften edges slightly
        occluded = cv2.GaussianBlur(
            occluded,
            (5, 5),
            0
        )
        return occluded
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")

def apply_degradation_and_preprocess(path: str, corruption_type = None, severity: int = None, size: int = 224) -> np.ndarray:
    '''
    Input: numpy array image represenation
    Apply a random corruption of random severity
    Apply preprocessing to this degraded image - ie ready for model input
    Output: numpy array image represenation
    '''
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Couldnt read from {path}")

    if severity is None:
        severity = random.randint(1, 5)

    degraded = apply_degradation(image, corruption_type, severity)

    processed = preprocess_image_array(degraded, size)

    return processed