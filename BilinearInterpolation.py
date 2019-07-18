import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

im = Image.open("download.jpg")
np_im = np.array(im, dtype=np.uint8)
# test = Image.fromarray(np_im)
# test.show()
# print(type(np_im.shape))
# print(np_im[0].size)
# print(4*np.asarray(np_im.shape))
# print((np_im.shape[2]))
#
# print(np_im[120, 120, 0])
# # np.array(())


def bilinearInterpolation(image, sizeFactor):
    """
    So I am going to go through each pixel in the original size image then im going to look at each pixel  to the right
    and to the bottom. Looking at these 4 pixels i will create a grid of pixel values the size of this grid will be
    the size factor.
    My loop will be look through each channel then the columns and rows. the loop will also go through the grid and do
    the bilinear interpolation
    Bilinear interpolation will look at the area of the grid and compare it to the value of the orignal image to
    determine the value of the pixel in the grid.
    Sorry to anyone reading this. These are just my unpolished thoughts
    :param image: numpy array image
    :param sizeFactor: how much you want to the size increase by integers only (for now)
    :return: enlarged image
    """
    # print(sizeFactor*np.asarray(image.shape[:2]))
    arr = sizeFactor*np.asarray(image.shape[:2])
    arr = np.append(arr, image.shape[2])
    out = np.zeros(arr, dtype=np.uint8) #np.zeros(sizeFactor*np.asarray(image.shape[:2]), image.shape[2])
    for k in range(image.shape[2]):  # Channels
        for i in range(image.shape[0]):  # Columns/x
            for j in range(image.shape[1]):  # Rows/y
                v1 = image[i, j, k]/255
                if i + 1 < image[0].size:
                    v2 = v1
                else:
                    v2 = image[i + 1, j, k]/255
                if j + 1 < image[1].size:
                    v3 = v1
                else:
                    v3 = image[i, j + 1, k]/255
                if i + 1 < image[1].size and j + 1 < image[1].size:
                    v4 = v1
                else:
                    v4 = image[i + 1, j + 1, k] /255
                for x in range(sizeFactor):
                    for y in range(sizeFactor):
                        d1 = sizeFactor - x
                        d2 = abs(sizeFactor - d1)
                        d3 = sizeFactor - y
                        d4 = abs(sizeFactor - d3)
                        q1 = (v1*d2 + v2*d1) /sizeFactor
                        q2 = (v3*d2 + v4*d1) /sizeFactor
                        q = (q1*d4 + q2*d3) /sizeFactor
                        a = i*sizeFactor + x
                        b = j*sizeFactor + y
                        c = k
                        out[a, b, c] = q * 255
    return out


out = bilinearInterpolation(np_im, 3)
resized = Image.fromarray(out)
resized.save('resized.png')
resized.show()
print(np_im)
print(out)
print(np_im.shape)
print(out.shape)