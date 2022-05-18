from PIL import ImageFile
from PIL import Image
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image