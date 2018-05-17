import numpy as np
from shape import Shape
from pca_analysis import run_pca
from utils.plot import plot
from utils.gpa import gpa
from myPCA import pca

def pdm(r):

    images, landmarks, landmarks_per_image = r
    # perform GPA
    mean, aligned = gpa(np.asarray(landmarks))
    plot('gpa', mean, aligned)

    # perform PCA
    eigenvalues, eigenvectors, m = pca(aligned)
    plot('eigenvectors', m, eigenvectors)

    # create PointDistributionModel instance
    model = PointDistributionModel(eigenvalues, eigenvectors, m)
    plot('deformablemodel', model)

    return model


class PointDistributionModel(object):
    '''
    Model created based on a mean image and a matrix
    of eigenvectors and corresponding eigenvalues.
    Based on shape parameters, it is able to create a
    variation on the mean shape.

    Eigenvectors are scaled according to Blanz p.2 eq.7.
    '''
    def __init__(self, eigenvalues, eigenvectors, mean):
        self.dimension = eigenvalues.size
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.mean = Shape(mean)

        # create a set of scaled eigenvectors
        self.scaled_eigenvectors = np.dot(self.eigenvectors, np.diag(self.eigenvalues))

    def deform(self, shape_param):
        '''
        Reconstruct a shape based on principal components and a set of
            parameters that define a deformable model (see Cootes p. 6 eq. 2)

        in: Tx1 vector deformable model b
        out: 1xC deformed image
        '''
        return Shape(self.mean.vector + self.scaled_eigenvectors.dot(shape_param))
