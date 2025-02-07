import numpy as np
from scipy.special import gamma

def knn_rrt(dimension):
    k_opt = 2 ** (dimension + 1)
    k_opt *= np.e * (1 + (1 / dimension))
    return int(np.ceil(k_opt))

def knn_prm(dimension):
    k_opt = np.e * (1 + (1 / dimension))
    return int(np.ceil(k_opt))

def unit_ball_volume(dimension):
    return (np.pi ** (dimension / 2)) / gamma((dimension / 2) + 1)

def s1_volume():
    return 2 * np.pi

def s2_volume():
    return 4 * np.pi

def so3_volume():
    return 16 * np.pi ** 2

# Note: the optimal radii are given in terms of the volume of the collision-free
# configuration space. Since we don't know this value, we instead use the volume
# of the whole configuration space, which leads to a higher radius. This ensures
# asymptotic optimality, but it's potentially not the smallest radius which
# would work.

# Because c_volume is strictly larger than the volume of c_free (unless
# there are zero obstacles), we don't have to add epsilon to handle the
# strictness of the bound.

def radius_rrt(dimension, c_volume):
    rad_opt = (2 * (1 + (1 / dimension))) ** (1 / dimension)
    rad_opt *= (c_volume / unit_ball_volume(dimension)) ** (1 / dimension)
    return rad_opt

def radius_prm(dimension, c_volume):
    rad_opt = 2 * ((1 + (1 / dimension)) ** (1 / dimension))
    rad_opt *= (c_volume / unit_ball_volume(dimension)) ** (1 / dimension)
    return rad_opt