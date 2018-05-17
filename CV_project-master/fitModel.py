import numpy as np
from alignModel import Aligner

from shape import Shape

class FitModel(object):
    def __init__(self,model):
       self.pdmodel = model
       self.aligner = Aligner()
       print("align .........\n",self.aligner)
       self.start_pose = ()



    def fit(self, prev_shape, new_shape, mode=0, n=None):
        '''
        Algorithm that finds the best shape parameters that match identified
        image points.

        In: Model instance ,
            array of new image points (x1, x2, ..., xN, y1, y2,..., yN)
        Out: the pose params (Tx, Ty, s, theta) and shape parameter (c) to
            fit the model to the image
        '''
        if not isinstance(new_shape, Shape):
            new_shape = Shape(new_shape)
        if not isinstance(prev_shape, Shape):
            prev_shape = Shape(prev_shape)
        if not self.start_pose:
            raise ValueError('No inital pose parameters found.')

        # find pose parameters to align with new image points
        Tx, Ty, s, theta = self.start_pose
        dx, dy, ds, dTheta = self.aligner.get_pose_parameters(prev_shape, new_shape)
        changed_pose = (Tx + dx, Ty + dy, s*(1+ds), theta+dTheta)

        # align image with model
        y = self.aligner.invert_transform(new_shape, changed_pose)

        # SVD on scaled eigenvectors of the model
        u, w, v = np.linalg.svd(self.pdmodel.scaled_eigenvectors, full_matrices=False)
        W = np.zeros_like(w)

        # define weight vector n
        if n is None:
            last_eigenvalue = self.pdmodel.eigenvalues[-1]
            n = last_eigenvalue**2 if last_eigenvalue**2 >= 0 else 0

        # calculate the shape vector
        W = np.diag(w/((w**2) + n))
        c = (v.T).dot(W).dot(u.T).dot(y.vector)

        return changed_pose, c
