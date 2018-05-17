import cv2
import numpy as np
def display_image(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.resizeWindow('image', 800, 600)

    cv2.waitKey()

def display_images(list_img):
    h_concat = []
    for i, h_images in enumerate(list_img):
        h_concat.append(np.concatenate(h_images, axis=1))
    all_concat = np.concatenate(h_concat, axis=0)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', all_concat)
    cv2.resizeWindow('image', 1600, 1200)
    cv2.waitKey()
    # rows = int(len(list_img) / 2)+1
    # i=1
    # for img in list_img:
    #     plt.subplot(rows, 2, i), plt.imshow(img, cmap='gray')
    #     i+=1
    # plt.show()


def resize(image, width, height):
    """Resizes the given image to the given width and height.
    Args:
        image: The radiograph cvto resize.
        width (int): The new width for the image.
        height (int): The new height for the image.
    Returns:
        The given image resized to the given width and height, and the scaling factor.
    """
    #find minimum scale to fit image on screen
    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale))), scale
