#!/opt/apps/intel18/python3/3.7.0/bin/python3

import sys
import numpy as np

filename = str(sys.argv[1])
ops = int(sys.argv[2])

# ignore first column in 2-points, first 3 columns in 3-points and 4-points
ignore = 1
if (ops > 2): ignore = 3

A = np.loadtxt("../results/" + filename)[:,ignore:]
B = np.loadtxt("../test-results/" + filename)[:,ignore:]

error = np.linalg.norm((A-B)/B)

print("%e: %s" % (error, filename))
