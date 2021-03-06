# This file contains all functions used to convert between coordinate systems
import numpy as np

# Quaternion to Rotation Matrix
def quaternion_to_r (q):
    R = np.zeros((3,3))
    R[0,0] = 1 - 2*q[2]*q[2] - 2*q[3]*q[3]
    R[0,1] = 2*q[1]*q[2] - 2*q[0]*q[3]
    R[0,2] = 2*q[1]*q[3] + 2*q[0]*q[2]
    R[1,0] = 2*q[1]*q[2] + 2*q[0]*q[3]
    R[1,1] = 1 - 2*q[1]*q[1] - 2*q[3]*q[3]
    R[1,2] = 2*q[2]*q[3] - 2*q[0]*q[1]
    R[2,0] = 2*q[1]*q[3] - 2*q[0]*q[2]
    R[2,1] = 2*q[2]*q[3] + 2*q[0]*q[1]
    R[2,2] = 1 - 2*q[2]*q[2] - 2*q[1]*q[1]
    return R