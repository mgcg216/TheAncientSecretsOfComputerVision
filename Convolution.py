import numpy as np
from PIL import Image
import math

im = Image.open("resized.png")
np_im = np.array(im, dtype=np.uint8)

def Convolution(image, convFilter):
    """
    So I will go through each pixel of an image and use a convolution filter. The convultion filter will be an array
    that should be and odd box number like 3x3 5x5 etc. i will pad the edges with the same value of the pixel just
    for ease of programming.
    Im using a one Dimensional array because its faster I hope
    :param image: numpy image array
    :param convFilter: array of the filter must be even box size
    :return: output numpy image
    """
    startDistance = math.sqrt(len(convFilter))//2
    # convFilter = convFilter.reverse()
    if math.sqrt(len(convFilter))%2 != 1:
        print("Not Valide filter size")
        return
    length = int(math.sqrt(len(convFilter)))
    arr = np.asarray(image.shape)
    out = np.zeros(arr, dtype=np.uint8)
    for k in range(arr[2]):  # Channels
        for i in range(arr[0]):  # Columns/x
            for j in range(arr[1]):  # Rows/y
                sx = i-startDistance
                sy = j-startDistance
                temp = 0
                for y in range(length):
                    for x in range(length):
                        # if i < 0 or j < 0 or i
                        deltaX = sx + x
                        deltaY = sy + y
                        if deltaX < 0 or deltaY < 0 or deltaX >= arr[0] or deltaY >= arr[1]:
                            pixel = image[i, j, k]
                            filter = convFilter[length * y + x]
                            value = pixel * filter
                        else:
                            pixel = image[int(deltaX), int(deltaY), k]
                            filter = convFilter[length * y + x]
                            value = pixel * filter
                        # print("pixel: ", pixel, " fileter: ", filter)
                        temp = temp + value / len(convFilter)
                #         print("+", value)
                # print("===", temp)
                out[i, j, k] = temp
    return out

filter = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
filter2 = [0, -1, 0, -1, 5, -1, 0, -1, 0]
filter3 = [1, 1, 1, 0, 0, -1, 0, 0, 1]

out = Convolution(np_im, filter2)
conv = Image.fromarray(out)
conv.save('conv.png')
conv.show()
print(np_im)
print(out)
print(np_im.shape)
print(out.shape)
print(np_im)
