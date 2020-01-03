from MatrixMethods import *
from Reader import Reader as rd
from Camera import Camera as cam
from SingleImage import SingleImage
from ImagePair import ImagePair
import numpy as np
import PhotoViewer as pv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # lab 7
    pixel_size = 0.0024  # [mm]
    focal = 4241.656 * pixel_size
    # focal = 4241.656
    xp = 2585.205 * pixel_size
    yp = 1852.701 * pixel_size
    k1 = -0.0823
    k2 = -0.694
    p1 = -0.0279
    p2 = -0.0058
    # lhfeiwes
    # radial_distortions
    rd1 = np.array([0, k1, k2])
    # decentering distortions
    dd = np.array([0, p1, p2])
    # principal point
    principal_point = np.array([xp, yp])

    fiducialsCam = 'no fiducials'

    # create camera instance
    cam1 = cam(focal, principal_point, rd1, dd, fiducialsCam)

    # create SingleImage instances
    IMG_2003 = SingleImage(cam1)
    IMG_2004 = SingleImage(cam1)
    IMG_2005 = SingleImage(cam1)

    # points in images
    IMG_2003_points = rd.ReadSampleFile('IMG_2003.json')
    IMG_2004_points = rd.ReadSampleFile('IMG_2004.json')
    IMG_2005_points = rd.ReadSampleFile('IMG_2005.json')

    # Compute Inner Orientation
    inner_param_2003 = IMG_2003.ComputeInnerOrientation([])
    inner_param_2004 = IMG_2004.ComputeInnerOrientation([])
    inner_param_2005 = IMG_2005.ComputeInnerOrientation([])

    # image to camera
    cam_2003_points = IMG_2003.ImageToCamera(IMG_2003_points)
    cam_2004_points = IMG_2004.ImageToCamera(IMG_2004_points)
    cam_2005_points = IMG_2004.ImageToCamera(IMG_2005_points)

    # create ImagePair instances
    img_pair1 = ImagePair(IMG_2003, IMG_2004)
    img_pair2 = ImagePair(IMG_2004, IMG_2005)

    # relative Orientation
    relativeOrientationImage2004, SigmaX_1 = img_pair1.ComputeDependentRelativeOrientation(IMG_2003_points[:11],
                                                                                           IMG_2004_points[:11],
                                                                                           np.array(
                                                                                               [[1, 0, 0, 0, 0, 0]]))
    relativeOrientationImage2005, SigmaX_2 = img_pair2.ComputeDependentRelativeOrientation(IMG_2004_points[:11],
                                                                                           IMG_2005_points[:11],
                                                                                           np.array(
                                                                                               [[1, 0, 0, 0, 0, 0]]))

    # points in model system
    model1_points, v1 = img_pair1.ImagesToModel(IMG_2003_points, IMG_2004_points, 'geometric')
    model2_points, v2 = img_pair1.ImagesToModel(IMG_2004_points, IMG_2005_points, 'vector')
    # model1_points, v1 = img_pair1.ImagesToModel(IMG_2003_points, IMG_2004_points, 'geometric')
    # model2_points, v2 = img_pair1.ImagesToModel(IMG_2004_points, IMG_2005_points, 'geometric')

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img_pair1.drawImagePair(model1_points, ax)
    img_pair1.drawModel(model1_points, ax)

    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    # ax2.set_aspect('equal')
    # ax2 = fig2.add_subplot(111, projection='3d')
    img_pair1.drawModel(model2_points, ax2)

    plt.show()