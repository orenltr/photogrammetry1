import numpy as np
from ImagePair import ImagePair
from SingleImage import SingleImage
from Camera import Camera

class ImageTriple(object):
    def __init__(self, imagePair1, imagePair2):
        """
        Inisialize the ImageTriple class

        :param imagePair1: first image pair
        :param imagePair2: second image pair

        .. warning::

            Check if the relative orientation is solved for each image pair
        """
        self.__imagePair1 = imagePair1
        self.__imagePair2 = imagePair2

    def ComputeScaleBetweenModels(self, cameraPoint1, cameraPoint2, cameraPoint3):
        """
        Compute scale between two models given the relative orientation

        :param cameraPoints1: camera point in first camera space
        :param cameraPoints2: camera point in second camera space
        :param cameraPoints3:  camera point in third camera space

        :type cameraPoints1: np.array 1x3
        :type cameraPoints2: np.array 1x3
        :type cameraPoints3: np.array 1x3

        :return scale between the two models
        :rtype: float


        .. warning::

            This function is empty, need implementation
        """

        R1 = self.__imagePair1.RotationMatrix_Image1
        R12 = self.__imagePair1.RotationMatrix_Image2
        R23 = self.__imagePair2.RotationMatrix_Image2

        v1 = np.dot(R1, cameraPoint1.T)
        v2 = np.dot(R12, cameraPoint2.T)
        v3 = np.dot(np.dot(R12,R23), cameraPoint3.T)
        v1_normelized = v1 / np.linalg.norm(v1)
        v2_normelized = v2 / np.linalg.norm(v2)
        v3_normelized = v3 / np.linalg.norm(v3)
        d1 = np.cross(v1_normelized, v2_normelized)
        A1 = np.array([v1_normelized, d1, -v2_normelized]).T
        x1 = np.dot(np.linalg.inv(A1), self.__imagePair1.PerspectiveCenter_Image2)
        d2 = np.cross(v2_normelized, v3_normelized)
        R2b2 = np.dot(R12, self.__imagePair2.PerspectiveCenter_Image2)
        A2 = np.vstack((R2b2.T, v3_normelized, -d2)).T
        x2 = np.dot(np.linalg.inv(A2), np.dot(x1[2, 0], v2_normelized))
        scale = x2[0]
        return scale

    def RayIntersection(self, cameraPoints1, cameraPoints2, cameraPoints3):
        """
        Compute coordinates of the corresponding model point

        :param cameraPoints1: points in camera1 coordinate system
        :param cameraPoints2: points in camera2 coordinate system
        :param cameraPoints3: points in camera3 coordinate system

        :type cameraPoints1 np.array nx3
        :type cameraPoints2: np.array nx3
        :type cameraPoints3: np.array nx3

        :return: point in model coordinate system
        :rtype: np.array nx3


        """

        scale = self.ComputeScaleBetweenModels(cameraPoints1[0],cameraPoints2[0],cameraPoints3[0])

        R1 = self.__imagePair1.RotationMatrix_Image1
        R2 = self.__imagePair1.RotationMatrix_Image2
        R3 = np.dot(R2, self.__imagePair2.RotationMatrix_Image2)
        O1 = self.__imagePair1.PerspectiveCenter_Image1
        O2 = self.__imagePair1.PerspectiveCenter_Image2
        O3 = O2 + scale * np.dot(R2,self.__imagePair2.PerspectiveCenter_Image2)

        X = np.zeros([len(cameraPoints1), 3])  # optimal point
        # sigma = np.zeros([len(cameraPoints1), 3])

        dO = O2 - O1

        for i in range(len(cameraPoints1)):
            # compute rays
            v1 = np.dot(R1, cameraPoints1[i].T)
            v2 = np.dot(R2, cameraPoints2[i].T)
            v3 = np.dot(R3, cameraPoints3[i].T)
            v1_normelized = v1 / np.linalg.norm(v1)
            v2_normelized = v2 / np.linalg.norm(v2)
            v3_normelized = v3 / np.linalg.norm(v3)

            # adjustment
            A = np.vstack((np.eye(3) - np.outer(v1_normelized, v1_normelized.T),
                           np.eye(3) - np.outer(v2_normelized, v2_normelized.T),
                           np.eye(3) - np.outer(v3_normelized, v3_normelized.T)
                           ))
            l = np.vstack((A[0:3].dot(O1), A[3:6].dot(O2),A[6:].dot(O3)))
            X[i] = (np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(l)).T
            V = np.dot(A, X[i].T) - l[:, 0]
            # sigma[i] = (V.T*V)/(6-3)
        return X


    def drawModles(self, imagePair1, imagePair2, modelPoints1, modelPoints2,ax):
        """
        Draw two models in the same figure

        :param imagePair1: first image pair
        :param imagePair2:second image pair
        :param modelPoints1: points in the firt model
        :param modelPoints2:points in the second model
        :param ax: axes of the plot

        :type modelPoints1: np.array nx3
        :type modelPoints2: np.array nx3

        :return: None

        .. warning::
            This function is empty, need implementation
        """
        imagePair1.drawImagePair(modelPoints1,ax)
        imagePair2.drawImagePair(modelPoints2,ax)
        imagePair2.drawModel(modelPoints1,ax)
        imagePair2.drawModel(modelPoints2,ax)

# if __name__ == '__main__':
#     camera = Camera(152, None, None, None, None)
#     image1 = SingleImage(camera)
#     image2 = SingleImage(camera)
#     image3 = SingleImage(camera)
#     imagePair1 = ImagePair(image1, image2)
#     imagePair2 = ImagePair(image2, image3)
#     imageTriple1 = ImageTriple(imagePair11, imagePair22)
