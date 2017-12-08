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


def show_binary(name='processing', image=None):
    image = np.copy(image)
    image = image.astype(np.uint8)

    image[image > 0] = 255
    show(name, image)


def to_binary(image):
    image = np.copy(image)
    image = image.astype(np.uint8)

    image[image > 0] = 255
    return image

def me(path):
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


    image_rgb = cv2.imread(path, cv2.IMREAD_COLOR)
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

    #from skimage import transform, exposure

    #rows, cols = image.shape
    # image = transform.resize(image, (int(rows * .25), int(cols * .25)))
    #show_binary('processing', image)
    return image

def findTrafficSign(image):
    '''
    This function find blobs with blue color on the image.
    After blobs were found it detects the largest square blob, that must be the sign.
    '''

    # define range HSV for blue color of the traffic sign
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([110, 255, 255])
    rawCapture = image




    # At this point the image is available as stream.array
    frame = rawCapture

    frameArea = frame.shape[0] * frame.shape[1]

    # convert color image to HSV color scheme
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define kernel for smoothing
    kernel = np.ones((3, 3), np.uint8)
    # extract binary image with active blue regions
    mask = frame
    show_binary(image=mask)
    # morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    show_binary(image=mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    show_binary(image=mask)

    mask = mask.astype(np.uint8)

    #im2, cnt, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #show_binary(image=im2)
    # find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # defite string variable to hold detected sign description
    detectedTrafficSign = None

    # define variables to hold values during loop
    largestArea = 0
    largestRect = None

    # only proceed if at least one contour was found
    biggest_cnt = None
    if len(cnts) > 0:
        for cnt in cnts:
            # Rotated Rectangle. Here, bounding rectangle is drawn with minimum area,
            # so it considers the rotation also. The function used is cv2.minAreaRect().
            # It returns a Box2D structure which contains following detals -
            # ( center (x,y), (width, height), angle of rotation ).
            # But to draw this rectangle, we need 4 corners of the rectangle.
            # It is obtained by the function cv2.boxPoints()
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # count euclidian distance for each side of the rectangle
            sideOne = np.linalg.norm(box[0] - box[1])
            sideTwo = np.linalg.norm(box[0] - box[3])
            # count area of the rectangle
            area = sideOne * sideTwo
            # find the largest rectangle within all contours
            if area > largestArea:
                largestArea = area
                largestRect = box
                biggest_cnt = cnt


    if largestArea > frameArea * 0.02:
        # draw contour of the found rectangle on  the original image
        x, y, w, h = cv2.boundingRect(biggest_cnt)
        roi = frame[y:y + h, x:x + w]
        cv2.imshow("contour",roi)
        return roi
    else :
        return None


findTrafficSign(me('C:\\Users\\sss\\PycharmProjects\\EyeDrive\\test\\images\\6.jpg'))
cv2.waitKey(0)