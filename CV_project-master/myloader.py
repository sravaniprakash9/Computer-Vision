import os
import cv2
import numpy as np

from shape import Shape


IMAGE_DIR = 'C:/Users/srava/OneDrive/CV/ActiveShapeModel/Dental/dentalvision-master/dentalvision/_Data/Radiographs/'
IMAGE_AMOUNT = 14
IMAGE_DIM = (3023, 1597)

LANDMARK_DIR = 'C:/Users/srava/OneDrive/CV/ActiveShapeModel/Dental/dentalvision-master/dentalvision/_Data/Landmarks/original/'
LANDMARK_AMOUNT = 40            # amount of landmarks per tooth


class DataLoad(object):
    '''
    This class provides methods to load specific landmark datasets
    for training and testing. It loads images and landmarks from
    directory paths specified in constants IMAGE_DIR and LANDMARK_DIR.
    '''
    def __init__(self):
        self.images = self._load_grayscale_images()
        self.landmarks_per_image = self._load_landmarks_per_image()


    def leave_one_out(self, test_index=0):
        '''
        Divides into training and test sets by leaving one image and its
        landmarks out of the training set.

        in: int test_index; index to divide training/test
        out: np array images; array with grayscaled images per row
            np array landmarks; array with all landmarks as rows
            list of np arrays landmarks_per_image; array with rows of landmarks
                for each image
        '''
        training_images = np.asarray(self.images[:test_index] + self.images[test_index+1:])
        test_images = self.images[test_index]

        # create landmark training and test sets
        training_landmarks_per_image = np.vstack((self.landmarks_per_image[:test_index], self.landmarks_per_image[test_index+1:]))

        training_landmarks = np.vstack(training_landmarks_per_image[:][:])
        test_landmarks = np.vstack(self.landmarks_per_image[test_index][:])

        # compile training and test sets
        training_set = [training_images, training_landmarks, training_landmarks_per_image]
        test_set = [test_images, test_landmarks]

        return training_set, test_set

    def _load_grayscale_images(self):
        '''
        Load the images dataset.
        '''
        images = []
        for i in os.listdir(IMAGE_DIR):
            if i.endswith('.tif'):
                path = IMAGE_DIR + i
                images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
        return images

    def _load_landmarks_per_image(self):
        '''
        Compile landmarks per image for convenience in grayscale level
        training. This training phase needs an accurate relation between
        the images and their corresponding landmarks.

        Needs to be run after _load_grayscale_images()!
        '''
        if not self.images:
            raise IOError('Images have not been loaded yet.')

        landmarks_per_image = []
        for i in range(len(self.images)):
            # search for landmarks that include reference to image in path

            lms1 = [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-5.txt') in s]
            lms2 = [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-6.txt') in s]
            lms3= [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-7.txt') in s]
            lms4= [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-8.txt') in s]

            l = lms1[0].merge(lms2[0])
            l1=lms3[0].merge(lms4[0])
            lN=l.merge(l1)

            #ums1 = [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-5.txt') in s]
            #ums2 = [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-6.txt') in s]
            #ums3= [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-7.txt') in s]
            #ums4= [self._parse(LANDMARK_DIR + s) for s in os.listdir(LANDMARK_DIR) if ('landmarks'+str(i+1)+'-8.txt') in s]

            #u = ums1[0].merge(ums2[0])
            #u1=ums3[0].merge(ums4[0])
            #uN=u.merge(u1)

            landmarks_per_image.append([lN.vector])
            #landmarks_per_image.append([uN.vector])

        return np.asarray(landmarks_per_image)

    def _parse(self, path):
        '''
        Parse the data from path directory and return arrays of x and y coordinates
        Data should be in the form (x1, y1)

        in: String pathdirectory with list of landmarks (x1, y1, ..., xN, yN)
        out: 1xc array (x1, ..., xN, y1, ..., yN)
        '''
        data = np.loadtxt(path)
        x = np.absolute(data[::2, ])
        y = data[1::2, ]
        return Shape(np.hstack((x, y)))
