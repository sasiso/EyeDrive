import cv2
import numpy as np
from pytesser3 import *
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.segmentation import clear_border


class DetectSpeedLimit():
    def __init__(self):
        pass

    def detect(self, image):
        pass


def correct_intensity_df(image, image_hist_center=25.0):
    image_hist = exposure.histogram(image, 5)
    hist_values, hist_bins = image_hist

    # now take the max
    hist_max_ind = np.argmax(hist_values)
    bgIntensity = hist_bins[hist_max_ind]

    intensity_scale = image_hist_center / (255 * bgIntensity)

    corrected_image = intensity_scale * image
    corrected_image[corrected_image > 1] = 1
    corrected_image[corrected_image < 0] = 0

    return corrected_image


def show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(1)


def show_binary(name, image):
    image = np.copy(image)
    image = image.astype(np.uint8)

    image[image > 0] = 255
    show(name, image)


def to_binary(image):
    image = np.copy(image)
    image = image.astype(np.uint8)

    image[image > 0] = 255
    return image


image_rgb = cv2.imread('C:\\Users\\sss\\PycharmProjects\\EyeDrive\\test\\images\\6.jpg', cv2.IMREAD_COLOR)
show('original', image_rgb)

reddish = image_rgb[:, :, 0] > 160
show('processing', image_rgb)

image_rgb[reddish] = [255, 255, 255]
reddish = np.invert(reddish)
image_rgb[reddish] = [0, 0, 0]
image = image_rgb.astype(np.uint8)
image = rgb2gray(image_rgb)
show('processing', image)

image_copy = np.copy(image)

thresh = threshold_otsu(image)
bw = closing(image > thresh, square(2))
show_binary('processing', bw)

# remove artifacts connected to image border
cleared = clear_border(bw, buffer_size=10)
show_binary('processing', cleared)
# label image regions
label_image = label(cleared)
show_binary('processing', label_image)
image = to_binary(label_image)

from skimage import transform, exposure

rows, cols = image.shape
image = transform.resize(image, (int(rows * .25), int(cols * .25)))
show_binary('processing', image)

cv2.waitKey(1)
cv2.waitKey(0)
cv2.destroyAllWindows()
