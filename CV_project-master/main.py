from data_loader import RadioGraph
import pca_analysis
import utils.preprocess_image as pre
from pointdistributionmodel import pdm
from activeShapeModel import ActiveShapeModel
from myloader import DataLoad

import cv2
import manual_init as manual
import numpy as np
from utils import display
from landmarks import Landmarks
import leaveOneOut as l
from utils.plot import plot


if __name__ == '__main__':

    #img = cv2.imread('./_Data/Radiographs/02.tif')
    #display.display_image(img)
    #r=RadioGraph("./_Data")
    radiographs = [RadioGraph("./_Data", i) for i in range(1, 15)]

    data=DataLoad()
    training,test=data.leave_one_out(test_index=0)
    #training_set,test=l.leave_one_out(radiographs)

    # Model Setup
    #val,vec,mean=pca_analysis.run_pca(radiographs)
    #print("vec..",vec)


    #activeShape=ActiveShapeTraining(mean,val,vec)
    pdmodel = pdm(training)
    activeshape = ActiveShapeModel(pdmodel,3)








    #  pre.process_image_test(img)
    #l=Landmarks('./_Data/Landmarks/original/landmarks2-2.txt')
    #__init__('./_Data/Landmarks/original/landmarks2-2.txt')
  #  cent,l=manual.init(l,img)
   # print(cent)


    # radiographs = [RadioGraph("./.data", i) for i in range(1, 15)]
    # pca_analysis.run_pca(radiographs)
    ## Part 1: Build an Active Shape Model
    # We used the description fromthe original paper by Cootes et al.


    # Load the provided landmarks into your program

    # Preprocess  the  landmarks  to  normalize  translation,  rotation  and
    # scale differences (Procrustes Analysis?)

    # Analyze the data using a Principal Component Analysis (PCA), exposing
    # shape class variations

    # Analyze the obtained principal components

    ## Part 2 : Preprocess dental radiographs

    ## Part 3 : Fit the model to the image

    # First Guess

    # Iterative fitting


    pass
