import cv2
import numpy as np
import random

def apply_corruption(image: np.ndarray, corruption_type: str, severity: str) -> np.ndarray:
    '''
    Input: numpy array image represenation
    Apply a corruption of certain type, at a certain severity
     - with a degree of randomness to the level of which each corruption is applied
    Output: numpy array image represenation
    '''
    # if invalid parameters provided then return the same image
    if ((corruption_type not in ["occlusion", "noise"]) 
            and (severity not in ["low", "medium", "high"])):
        return image

    if corruption_type == "occlusion":
        severity_params = {
            "low": [],
            "medium": [],
            "high": [],
        }
        pass
    elif corruption_type == "noise":
        severity_params = {
            "low": [],
            "medium": [],
            "high": [],
        }
        img_float = image.astype(np.float64) / 255.0
        gauss = np.random.normal(1.0, severity_params[severity], img_float.shape)
        # apply noise, then clip between 0 and 1
        noisy_img = img_float * gauss
        noisy_img = np.clip(noisy_img, 0, 1)
        return (noisy_img * 255).astype(np.uint8)
    
    # code adapted from https://stackoverflow.com/a/57629531
    elif corruption_type == "blur":
        severity_params = {
            "low": {"kernel_size": random.randrange(5, 11, 2)},
            "medium": [],
            "high": [],
        }
        kernel_size = severity_params[severity]
        angle = random.randint(0, 180)

        matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int(kernel_size / 2), :] = 1
        kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
        kernel = kernel / np.sum(kernel)
        return cv2.filter2D(image, -1, kernel)