from MatrixMethods import *
from Reader import Reader as rd
from Camera import Camera as cam
from SingleImage import SingleImage
from ImagePair import ImagePair
from ImageTriple import ImageTriple
import numpy as np
import PhotoViewer as pv
from matplotlib import pyplot as plt

def drawModel(modelPoints,ax):
    """
    draws the rays to the modelpoints from the perspective center of the two images

    :param modelPoints: points in the model system [ model units]
    :param ax: axes of the plot
    :type modelPoints: np.array nx3
    :type ax: plot axes

    :return: none

    """

    ax.scatter(modelPoints[:11, 0], modelPoints[:11, 1], modelPoints[:11, 2], c='black', marker='^')
    ax.scatter(modelPoints[11:, 0], modelPoints[11:, 1], modelPoints[11:, 2], c='red', marker='o')

    x = modelPoints[:, 0]
    y = modelPoints[:, 1]
    z = modelPoints[:, 2]
    connectpoints(x, y, z, 12, 13)
    connectpoints(x, y, z, 13, 24)
    connectpoints(x, y, z, 12, 14)
    connectpoints(x, y, z, 13, 15)
    connectpoints(x, y, z, 15, 14)
    connectpoints(x, y, z, 15, 16)
    connectpoints(x, y, z, 14, 17)
    connectpoints(x, y, z, 16, 17)
    connectpoints(x, y, z, 15, 23)
    connectpoints(x, y, z, 24, 23)
    connectpoints(x, y, z, 16, 19)
    connectpoints(x, y, z, 17, 18)
    connectpoints(x, y, z, 22, 20)
    connectpoints(x, y, z, 20, 21)
    connectpoints(x, y, z, 20, 19)
    connectpoints(x, y, z, 19, 18)
    connectpoints(x, y, z, 18, 21)
    set_axes_equal(ax)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def connectpoints(x, y, z, p1, p2):

    x1, x2 = x[p1-1], x[p2-1]
    y1, y2 = y[p1-1], y[p2-1]
    z1, z2 = z[p1-1], z[p2-1]
    plt.plot([x1, x2], [y1, y2],[z1, z2], 'k-')

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
    model1_points, v1 = img_pair1.ImagesToModel(IMG_2003_points, IMG_2004_points, 'vector')
    model2_points, v2 = img_pair1.ImagesToModel(IMG_2004_points, IMG_2005_points, 'vector')
    # model1_points, v1 = img_pair1.ImagesToModel(IMG_2003_points, IMG_2004_points, 'geometric')
    # model2_points, v2 = img_pair1.ImagesToModel(IMG_2004_points, IMG_2005_points, 'geometric')

    # link models without scale adjustment
    R3 = np.dot(img_pair1.RotationMatrix_Image2, img_pair2.RotationMatrix_Image2)
    O3 = img_pair1.PerspectiveCenter_Image2 + np.dot(img_pair1.RotationMatrix_Image2,img_pair2.PerspectiveCenter_Image2)
    phi = np.arcsin(R3[0, 2])
    omega = np.arccos(R3[2, 2]/np.cos(phi))
    kappa = np.arccos(R3[0, 0]/np.cos(phi))

    #   create ImageTriple instance
    img_triple = ImageTriple(img_pair1,img_pair2)

    # calculate scale between models
    scale = np.zeros(len(model2_points))
    for i in range(len(model2_points)):
        scale[i] = img_triple.ComputeScaleBetweenModels(np.hstack((cam_2003_points[i], -cam1.focalLength)),
                                                         np.hstack((cam_2004_points[i], -cam1.focalLength)),
                                                         np.hstack((cam_2005_points[i], -cam1.focalLength)))
    scale_mean = np.mean(scale)
    scale_std = np.std(scale)

    # link models with scale adjustment
    R3 = np.dot(img_pair1.RotationMatrix_Image2, img_pair2.RotationMatrix_Image2)
    O3 = img_pair1.PerspectiveCenter_Image2 + scale_mean * np.dot(img_pair1.RotationMatrix_Image2,
                                                     img_pair2.PerspectiveCenter_Image2)
    phi = np.arcsin(R3[0, 2])
    omega = np.arccos(R3[2, 2] / np.cos(phi))
    kappa = np.arccos(R3[0, 0] / np.cos(phi))

    img_pair2.__relativeOrientationImage1 = np.array([[0, 0, 0, 0, 0, 0]]).T

    # points in model system
    model_triple_points = img_triple.RayIntersection(np.hstack((cam_2003_points, np.zeros((len(cam_2003_points),1)) -cam1.focalLength)),
                                                         np.hstack((cam_2004_points,np.zeros((len(cam_2004_points),1)) -cam1.focalLength)),
                                                         np.hstack((cam_2005_points, np.zeros((len(cam_2005_points),1))-cam1.focalLength)))






    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img_triple.drawModles(img_pair1,img_pair2,model1_points,model2_points , ax)
    # img_pair1.drawModel(model1_points, ax)
    #
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    img_pair1.drawModel(model1_points, ax2)
    img_pair2.drawModel(model2_points, ax2)

    fig3 = plt.figure()
    ax3 = fig3.gca(projection='3d')
    drawModel(model1_points, ax3)

    plt.show()