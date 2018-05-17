import numpy as np



def leave_one_out(image,landmarks, test_index=0):
        '''
        Divides into training and test sets by leaving one image and its
        landmarks out of the training set.

        in: int test_index; index to divide training/test
        out: np array images; array with grayscaled images per row
            np array landmarks; array with all landmarks as rows
            list of np arrays landmarks_per_image; array with rows of landmarks
                for each image
        '''
        training_images = np.asarray(image[:test_index] + image[test_index+1:])
        test_images = image[test_index]

        # create landmark training and test sets
        training_landmarks_per_image = np.vstack((landmarks[:test_index], landmarks[test_index+1:]))

        training_landmarks = np.vstack(training_landmarks_per_image[:][:])
        test_landmarks = np.vstack(landmarks[test_index][:])

        # compile training and test sets
        training_set = [training_images, training_landmarks, training_landmarks_per_image]
        test_set = [test_images, test_landmarks]

        return training_set, test_set
