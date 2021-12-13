import os
import numpy as np

image_dir = r'E:\models\scan4'
root=r'E:\models'
MRT = 0.7
#intrinsic parameters

K = np.array([
        [2892.33, 0, 823.205],
        [0, 2883.18,  619.071],
        [0, 0, 1]])

x = 0.5
y = 1