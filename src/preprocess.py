import cv2
import numpy as np

def preprocess_image(path: str, size = 224) -> np.ndarray:
    '''
    Input: path to read an image file, size for the output image dimensions ie (size x size pixels)
    Apply resizing, CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Output: numpy array image represenation
    '''
    # greyscale image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # resize to size
    img = cv2.resize(img, (size, size)) 

    # denoise
    img = cv2.fastNlMeansDenoising(img, h=12)

    # CLAHE good for xrays? https://pmc.ncbi.nlm.nih.gov/articles/PMC12784379/
    clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (8, 8)) # 8 x 8 default 
    img = clahe.apply(img)

    return img


def preprocess_image_array(img: np.ndarray, size: int = 224) -> np.ndarray:
    '''
    Apply preprocessing to an already loaded image
    '''
    # resize to size
    img = cv2.resize(img, (size, size)) 

    # denoise
    img = cv2.fastNlMeansDenoising(img, h=12)

    # CLAHE good for xrays? https://pmc.ncbi.nlm.nih.gov/articles/PMC12784379/
    clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (8, 8)) # 8 x 8 default 
    img = clahe.apply(img)

    return img