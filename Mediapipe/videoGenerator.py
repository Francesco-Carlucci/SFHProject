import cv2
import numpy as np
import glob

if __name__ == '__main__':
    img_array = []
    for filename in glob.glob('../GesRec/datasets/jester_kaggle/archive/Train/164/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('jester_turningHandCounterClockWise2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()