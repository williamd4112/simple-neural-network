import pandas as pd
import numpy as np
import random
import csv

import cv2

def one_hot(n, i):
    v = np.zeros(n)
    v[i] = 1
    return v

