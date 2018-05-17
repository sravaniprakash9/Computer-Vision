import numpy as np
from activeShapeModel import  ActiveShapeModel
from pointdistributionmodel import pdm

class ActiveShapeTraining(object):

    def __init__(self, radiographs, k=8, levels=4):
         # self.images, self.landmarks, self.landmarks_per_image = training_set
          self.radiographs=radiographs
          print '***Setting up Active Shape Model...'
          # 1. Train POINT DISTRIBUTION MODEL
          print '---Training Point-Distribution Model...'
          pdmodel = self.pointdistribution(self.radiographs)
         # 3. Train ACTIVE SHAPE MODEL
          print '---Initialising Active Shape Model...'
          self.activeshape = ActiveShapeModel(pdmodel)
          print 'done'

    def pointdistribution(self, landmarks):
        '''
        Create model of shape from input landmarks
        '''
        return pdm(landmarks)

