import numpy as np
import csv
import sys

with open(sys.argv[1], 'w') as f:
    K = 3
    N = int(sys.argv[2])
    I = np.identity(K).astype(np.int32)
    for k in xrange(K):
        yn = I[k, :]
        for i in xrange(N):
            f.write(','.join([str(yn_i) for yn_i in yn]) + '\n')
