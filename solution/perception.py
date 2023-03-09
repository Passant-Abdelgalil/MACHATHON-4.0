import cv2
import numpy as np
from types import tuple


def get_center_lane(image: np.ndarray) -> np.ndarray:
    """
    Get the center lane of the road by transforming the image to HSV then filter the white color
    Parameters
    ----------
    image : np.ndarray
        The image to process
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # filter non white color
    lower_red = np.array([0, 0, 190])
    upper_red = np.array([0, 3, 255])

    # filter by white color which is just the grey region of the road
    res = cv2.inRange(hsv, lower_red, upper_red)

    # get the contours
    contours, _ = cv2.findContours(
        res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # remove the small contours
    mask = np.zeros(res.shape, np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 28000:
            mask = cv2.drawContours(mask, [contour], -1, 255, -1)

    # remove the top part of the image
    mask[-70:, :] = 0
    return mask


def get_center(img: np.ndarray) -> tuple(int, int):
    """
    Get the center of the lane
    Parameters
    ----------
    img : np.ndarray
        The image to process
    """
    M = cv2.moments(img)
    cxm, cxy = None, None
    if M['m00'] > 0:
        cxm = int(M['m10']/M['m00'])
        cym = int(M['m01']/M['m00'])
        cv2.circle(img, (cxm, cym), 20, (0, 0, 0), -1)

    return cxm, cxy
