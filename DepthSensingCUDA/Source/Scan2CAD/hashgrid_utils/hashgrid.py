import sys
import os
import struct
import glob
import numpy as np
import math
import Vox
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as pl
import torch
import torch.nn.functional as sampling
VOXEL_SIZE = 8
VOXEL_SIZE_3 = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE
HASH_BUCKET_SIZE = 10
from ctypes import c_ulong
import h5py
import pickle
# TODO Change virtualvoxelsize
virtualVoxelSize = np.float32(0.030)
HashParam_SDFBlockSize = 8
s_SDFMarchingCubeThreshFactor = np.float32(10.0)
threshMarchingCubes = s_SDFMarchingCubeThreshFactor*virtualVoxelSize
threshMarchingCubes2 = threshMarchingCubes
m_minCorner = 0
m_maxCorner = 0
m_boxEnabled = 1
trunc = float(1.0)

edgeTable = [0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
             0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
             0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
             0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
             0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
             0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
             0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
             0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
             0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
             0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
             0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
             0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
             0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
             0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
             0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
             0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
             0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
             0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
             0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
             0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
             0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
             0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
             0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
             0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
             0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
             0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
             0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
             0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
             0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
             0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
             0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
             0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0]

triTable = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
                     [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
                     [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
                     [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
                     [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
                     [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
                     [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
                     [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
                     [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
                     [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
                     [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
                     [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
                     [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
                     [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
                     [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
                     [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
                     [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
                     [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
                     [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
                     [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
                     [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
                     [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
                     [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
                     [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
                     [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
                     [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
                     [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
                     [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
                     [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
                     [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
                     [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
                     [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
                     [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
                     [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
                     [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
                     [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
                     [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
                     [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
                     [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
                     [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
                     [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
                     [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
                     [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
                     [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
                     [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
                     [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
                     [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
                     [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
                     [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
                     [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
                     [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
                     [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
                     [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
                     [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
                     [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
                     [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
                     [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
                     [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
                     [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
                     [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
                     [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
                     [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
                     [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
                     [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
                     [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
                     [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
                     [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
                     [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
                     [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
                     [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
                     [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
                     [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
                     [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
                     [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
                     [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
                     [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
                     [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
                     [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
                     [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
                     [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
                     [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
                     [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
                     [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
                     [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
                     [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
                     [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
                     [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
                     [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
                     [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
                     [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
                     [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
                     [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
                     [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
                     [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
                     [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
                     [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
                     [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
                     [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
                     [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
                     [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
                     [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
                     [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
                     [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
                     [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
                     [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
                     [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
                     [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
                     [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
                     [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
                     [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
                     [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
                     [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
                     [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
                     [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
                     [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
                     [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
                     [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
                     [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
                     [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
                     [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
                     [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
                     [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
                     [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
                     [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
                     [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
                     [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
                     [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
                     [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
                     [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
                     [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
                     [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
                     [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
                     [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
                     [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
                     [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
                     [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
                     [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
                     [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
                     [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
                     [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
                     [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
                     [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
                     [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
                     [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
                     [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
                     [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
                     [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
                     [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
                     [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
                     [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
                     [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
                     [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
                     [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
                     [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
                     [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
                     [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
                     [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
                     [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
                     [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
                     [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
                     [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
                     [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
                     [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
                     [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
                     [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
                     [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
                     [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
                     [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
                     [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
                     [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
                     [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
                     [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
                     [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
                     [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
                     [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
                     [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
                     [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
                     [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
                     [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
                     [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
                     [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
                     [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
                     [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
                     [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]



class HashEntry:
    def __init__(self, pos=np.array([0, 0, 0], dtype=int), ptr=-1, offset=0):
        self.pos = pos
        if ptr == "FREE_ENTRY":
            self.ptr = -2
        else:
            self.ptr = ptr
        self.offset = c_ulong(offset)


class HashData:
    def __init__(self, SDFBlocks =None, entries = None):
        self.SDFBlocks = SDFBlocks
        self.entries = entries

    def getHashEntry(self, worldPos):
        blockID = self.worldToSDFBlock(worldPos)
        return self.getHashEntryForSDFBlockPos(blockID)

    #CHECKED
    def getHashEntryForSDFBlockPos(self, sdfBlock):
        h = self.computeHashPos(sdfBlock)
        #hp = h * HASH_BUCKET_SIZE
        entry = HashEntry(sdfBlock, -2, 0)
        for j in range(0, len(self.entries)):
            # i = j + hp
            curr = self.entries[j]
            if (curr.pos[0] == entry.pos[0] and curr.pos[1] == entry.pos[1] and curr.pos[2] == entry.pos[2]
                    and curr.ptr != -2):
                return curr, j
        return entry, -1

    def computeHashPos(self, virtualVoxelPos):
        p0 = 73856093
        p1 = 19349669
        p2 = 83492791
        res = ((virtualVoxelPos[0] * p0) ^ (virtualVoxelPos[1] * p1) ^ (virtualVoxelPos[2] * p2)) % HASH_BUCKET_SIZE
        if res < 0: res += HASH_BUCKET_SIZE
        return res

    #checked
    def getVoxel(self, worldPos):
        hashEntry, SDFIndex = self.getHashEntry(worldPos)
        v = Voxel(sdf=0, color=0, weight=0)
        if not hashEntry.ptr == -2:
            virtualVoxelPos = self.worldToVirtualVoxelPos(worldPos)
            index = self.virtualVoxelPosToLocalSDFBlockIndex(virtualVoxelPos)
            v = self.SDFBlocks[SDFIndex * 512+ index]
        return v

    def worldToSDFBlock(self, worldPos):
        return self.virtualVoxelPosToSDFBlock(self.worldToVirtualVoxelPos(worldPos))

    #CHECKED
    def virtualVoxelPosToSDFBlock(self, virtualVoxelPos):
        if virtualVoxelPos[0] < 0: virtualVoxelPos[0] -= VOXEL_SIZE - 1
        if virtualVoxelPos[1] < 0: virtualVoxelPos[1] -= VOXEL_SIZE - 1
        if virtualVoxelPos[2] < 0: virtualVoxelPos[2] -= VOXEL_SIZE - 1
        return np.array([virtualVoxelPos[0] / VOXEL_SIZE, virtualVoxelPos[1] / VOXEL_SIZE,
                         virtualVoxelPos[2] / VOXEL_SIZE], dtype=int)

    #CHECKED
    def worldToVirtualVoxelPos(self, pos):
        # TODO Change virtualvoxelsize
        p = pos / virtualVoxelSize
        return (p + np.array([np.sign(p[0]), np.sign(p[1]), np.sign(p[2])], dtype=np.float32) * np.float32(0.5)).astype(int)

    #Checked
    def virtualVoxelPosToLocalSDFBlockIndex(self, virtualVoxelPos):
        localVoxelPos = np.array([
            virtualVoxelPos[0] % VOXEL_SIZE,
            virtualVoxelPos[1] % VOXEL_SIZE,
            virtualVoxelPos[2] % VOXEL_SIZE], dtype=int)

        if (localVoxelPos[0] < 0): localVoxelPos[0] += VOXEL_SIZE
        if (localVoxelPos[1] < 0): localVoxelPos[1] += VOXEL_SIZE
        if (localVoxelPos[2] < 0): localVoxelPos[2] += VOXEL_SIZE

        return self.linearizeVoxelPos(localVoxelPos)

    def linearizeVoxelPos(self, virtualVoxelPos):
        return virtualVoxelPos[2] * VOXEL_SIZE * VOXEL_SIZE + virtualVoxelPos[1] * VOXEL_SIZE + virtualVoxelPos[0]

    def worldToVirtualVoxelPosFloat(self, pos):
        return (pos / virtualVoxelSize)

    def SDFBlockToVirtualVoxelPos(self, sdfBlock):
        return sdfBlock * VOXEL_SIZE

    def virtualVoxelPosToWorld(self, pos):
        return pos.astype(np.float32) * virtualVoxelSize

    def voxelPosToLocalVoxIndex(self, virtualVoxelPos, vox, minCorner):
        rotX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        virtualVoxelPos = rotX @ virtualVoxelPos
        localVoxelPos = np.array([
            virtualVoxelPos[0] % vox.dims[0],
            virtualVoxelPos[1] % vox.dims[1],
            virtualVoxelPos[2] % vox.dims[2]], dtype=int)

        if (localVoxelPos[0] < 0): localVoxelPos[0] += vox.dims[0]
        if (localVoxelPos[1] < 0): localVoxelPos[1] += vox.dims[1]
        if (localVoxelPos[2] < 0): localVoxelPos[2] += vox.dims[2]

        return self.linearizeVoxelPosVox(localVoxelPos,vox.dims, minCorner)

    def linearizeVoxelPosVox(self, virtualVoxelPos, voxelPosMin, dims, gt=None):
        vox_voxelPos = virtualVoxelPos - voxelPosMin
        y = vox_voxelPos[1]
        z = vox_voxelPos[2]
        vox_voxelPos[2] = y
        vox_voxelPos[1] = z
        for i in range(0,3):
            if vox_voxelPos[i] >= dims[i]:
                return None, None
        if gt != None:
            v = np.array2string(vox_voxelPos, precision=2, separator=',', suppress_small=True)
            if v not in gt:
                #print(v)
                return None, None
        voxIndex = vox_voxelPos[2] * dims[0] * dims[1] + vox_voxelPos[1] * dims[0] + vox_voxelPos[0]
        return voxIndex, vox_voxelPos


class Voxel:
    def __init__(self, sdf =None, color = None, weight = None, pos = None):
        self.sdf = sdf
        self.color = color
        self.weight = weight
        self.pos = None #grid pos


class ChunkGrid:
    def __init__(self, SDFBlockList=None, SDFBlockDescList=None):
        self.m_SDFBlocks = SDFBlockList
        self.m_ChunkDesc = SDFBlockDescList


class Triangle:
    def __init__(self, v0=None, v1=None, v2=None):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2


class Vertex:
    def __init__(self, p=None, c=None, w=None):
        self.p = p
        self.c = c
        self.w = w


class CUDASceneRepChunkGrid:
    def __init__(self, filename, indexing_type):
        assert os.path.isfile(filename), "file not found: %s" % filename
        self.indexing_type = indexing_type
        self.load_sample(filename)

    def load_sample(self, filename):
        fin = open(filename, 'rb')
        self.filename = filename
        self.hashGridVersion = struct.unpack('I', fin.read(4))[0]
        self.voxelSize = struct.unpack('f', fin.read(4))[0]
        self.voxelExtents = np.asarray(struct.unpack('f' * 3, fin.read(12)))
        self.gridDimensions = np.asarray(struct.unpack('i' * 3, fin.read(12)))
        self.minGridPos = np.asarray(struct.unpack('i' * 3, fin.read(12)))
        self.maxGridPos = np.asarray(struct.unpack('i' * 3, fin.read(12)))
        self.initialChunkListSize = struct.unpack('I', fin.read(4))[0]
        self.numOccupiedChunks = struct.unpack('I', fin.read(4))[0]
        #print("minGridPos" + str(self.minGridPos))
        #print("maxGridPos" + str(self.maxGridPos))
        #print("numOccupiedChunks: "+str(self.numOccupiedChunks))
        #print("voxelExtents: " + str(self.voxelExtents))

        # Chunk Loop
        if(self.indexing_type =='chunkGrid'):
            self.m_grid = [None] * self.gridDimensions[0] * self.gridDimensions[1] * self.gridDimensions[2]
        else:
            self.m_grid = list()
            self.gridIndexList = list()

        for i in range(0, self.numOccupiedChunks):
            # index 4 bytes
            gridIndex = struct.unpack('i', fin.read(4))[0]

            # size 8 bytes  total = 76 bytes
            size = int.from_bytes(fin.read(8), "little", signed=False)

            # SDF Block Loop
            SDFBlock = []
            for j in range(0, size):
                # Voxel LOOP
                for t in range(0, VOXEL_SIZE_3):
                    sdf = struct.unpack('f', fin.read(4))[0]
                    colorR = struct.unpack('B', fin.read(1))[0]
                    colorG = struct.unpack('B', fin.read(1))[0]
                    colorB = struct.unpack('B', fin.read(1))[0]
                    weight = struct.unpack('B', fin.read(1))[0]

                    # print("sdf: " + str(sdf) + "R: " + str(colorR) + "G: " + str(colorG) + "B: " + str(colorB) + "weight: " + str(weight))

                    SDFBlock.append(Voxel(sdf, np.array([colorR, colorG, colorB], dtype=int), weight))

            # size 8 bytes  total = 76 bytes
            size = int.from_bytes(fin.read(8), "little", signed=False)
            SDFBlockDesc = []

            # SDF Block Desc Loop
            #print("SDFBlockDesc: " + str(size))
            for j in range(0, size):
                # pos 12 bytes
                posX = struct.unpack('i', fin.read(4))[0]
                posY = struct.unpack('i', fin.read(4))[0]
                posZ = struct.unpack('i', fin.read(4))[0]
                ptr = struct.unpack('i', fin.read(4))[0]
                # print(str([posX, posY, posZ]))
                SDFBlockDesc.append(HashEntry(np.array([posX, posY, posZ], dtype=int), ptr, 0))

            if self.indexing_type == 'chunkGrid':
                self.m_grid[gridIndex] = HashData(SDFBlock, SDFBlockDesc)
            else:
                self.m_grid.append(HashData(SDFBlock, SDFBlockDesc))
                self.gridIndexList.append(self.delinearizeChunkIndex(gridIndex))


    def chunkToWorld(self, posChunk):
        return np.array([posChunk[0] * self.voxelExtents[0], posChunk[1] * self.voxelExtents[1],
                         posChunk[2] * self.voxelExtents[2]], dtype=np.float32)

    def containsSDFBlocksChunk(self, chunkPos):
        index = self.linearizeChunkPos(chunkPos)
        # return ((m_grid[index] != 0) && m_grid[index]->isStreamedOut());
        return (self.m_grid[index] != None)

    def getWorldPosChunk(self, chunkPos):
        res = np.zeros(3)
        res[0] = chunkPos[0] * self.voxelExtents[0]
        res[1] = chunkPos[1] * self.voxelExtents[1]
        res[2] = chunkPos[2] * self.voxelExtents[2]
        return res

    def trilinearInterpolationSimpleFastFast(self, pos, hashData):
        oSet = virtualVoxelSize
        posDual = pos - np.array((oSet / np.float32(2.0), oSet / np.float32(2.0), oSet / np.float32(2.0)))
        weight = frac(hashData.worldToVirtualVoxelPosFloat(pos))
        dist = np.float32(0.0)
        colorFloat = np.array([0, 0, 0], dtype=np.float32)

        v = hashData.getVoxel(posDual + np.array([0, 0, 0], dtype=np.float32))
        if v.weight == 0:
            return -1, -1, False
        vColor = np.array((v.color[0], v.color[1], v.color[2]), dtype=np.float32)
        dist += (np.float32(1.0) - weight[0]) * (np.float32(1.0) - weight[1]) * (np.float32(1.0) - weight[2]) * v.sdf
        colorFloat += (np.float32(1.0) - weight[0]) * (np.float32(1.0) - weight[1]) * (np.float32(1.0) - weight[2]) * vColor

        v = hashData.getVoxel(posDual + np.array([oSet, 0, 0], dtype=np.float32))
        if v.weight == 0:
            return -1, -1, False
        vColor = np.array((v.color[0], v.color[1], v.color[2]), dtype=np.float32)
        dist += (weight[0]) * (np.float32(1.0) - weight[1]) * (np.float32(1.0) - weight[2]) * v.sdf
        colorFloat += (weight[0]) * (np.float32(1.0) - weight[1]) * (np.float32(1.0) - weight[2]) * vColor

        v = hashData.getVoxel(posDual + np.array([0, oSet, 0], dtype=np.float32))
        if v.weight == 0:
            return -1, -1, False
        vColor = np.array((v.color[0], v.color[1], v.color[2]), dtype=np.float32)
        dist += (np.float32(1.0) - weight[0]) * (weight[1]) * (np.float32(1.0) - weight[2]) * v.sdf
        colorFloat += (np.float32(1.0) - weight[0]) * (weight[1]) * (np.float32(1.0) - weight[2]) * vColor

        v = hashData.getVoxel(posDual + np.array([0, 0, oSet], dtype=np.float32))
        if v.weight == 0:
            return -1, -1, False
        vColor = np.array((v.color[0], v.color[1], v.color[2]), dtype=np.float32)
        dist += (np.float32(1.0) - weight[0]) * (np.float32(1.0) - weight[1]) * (weight[2]) * v.sdf
        colorFloat += (np.float32(1.0) - weight[0]) * (np.float32(1.0) - weight[1]) * (weight[2]) * vColor

        v = hashData.getVoxel(posDual + np.array([oSet, oSet, 0], dtype=np.float32))
        if v.weight == 0:
            return -1, -1, False
        vColor = np.array((v.color[0], v.color[1], v.color[2]), dtype=np.float32)
        dist += (weight[0]) * (weight[1]) * (np.float32(1.0) - weight[2]) * v.sdf
        colorFloat += (weight[0]) * (weight[1]) * (np.float32(1.0) - weight[2]) * vColor

        v = hashData.getVoxel(posDual + np.array([0, oSet, oSet], dtype=np.float32))
        if v.weight == 0:
            return -1, -1, False
        vColor = np.array((v.color[0], v.color[1], v.color[2]), dtype=np.float32)
        dist += (np.float32(1.0) - weight[0]) * (weight[1]) * (weight[2]) * v.sdf
        colorFloat += (np.float32(1.0) - weight[0]) * (weight[1]) * (weight[2]) * vColor

        v = hashData.getVoxel(posDual + np.array([oSet, 0, oSet], dtype=np.float32))
        if v.weight == 0:
            return -1, -1, False
        vColor = np.array((v.color[0], v.color[1], v.color[2]), dtype=np.float32)
        dist += (weight[0]) * (np.float32(1.0) - weight[1]) * (weight[2]) * v.sdf
        colorFloat += (weight[0]) * (np.float32(1.0) - weight[1]) * (weight[2]) * vColor

        v = hashData.getVoxel(posDual + np.array([oSet, oSet, oSet], dtype=np.float32))
        if v.weight == 0:
            return -1, -1, False
        vColor = np.array((v.color[0], v.color[1], v.color[2]), dtype=np.float32)
        dist += (weight[0]) * (weight[1]) * (weight[2]) * v.sdf
        colorFloat += (weight[0]) * (weight[1]) * (weight[2]) * vColor

        return dist, colorFloat, True

    def vertexInterp(self, isolevel, p1, p2, d1, d2, c1, c2, w1, w2):
        if abs(isolevel - d1) < np.float32(0.00001):
            resPosition = (p1[0],p1[1],p1[2])
            resColor = (c1[0] / np.float32(255.0), c1[1] / np.float32(255.0), c1[2] / np.float32(255.0))
            resWeight = w1 / np.float32(255.0)
            return Vertex(resPosition, resColor, (np.float32(255),))
        if abs(isolevel - d2) < np.float32(0.00001):
            resPosition = (p2[0], p2[1], p2[2])
            resColor = (c2[0] / np.float32(255.0), c2[1] / np.float32(255.0), c2[2] / np.float32(255.0))
            resWeight = w2 / np.float32(255.0)
            return Vertex(resPosition, resColor, (np.float32(255),))
        if abs(d1 - d2) < np.float32(0.00001):
            resPosition = (p1[0], p1[1], p1[2])
            resColor = (c1[0] / np.float32(255.0), c1[1] / np.float32(255.0), c1[2] / np.float32(255.0))
            resWeight = w1 / np.float32(255.0)
            return Vertex(resPosition, resColor, (np.float32(255),))

        mu = (isolevel - d1) / (d2 - d1)
        resPosition = (p1[0] + mu * (p2[0] - p1[0]), p1[1] + mu * (p2[1] - p1[1]), p1[2] + mu * (p2[2] - p1[2]))
        resColor = (np.float32(c1[0] + mu * np.float32(c2[0] - c1[0])) , np.float32(c1[1] + mu * np.float32(c2[1] - c1[1])),
             np.float32(c1[2] + mu * np.float32(c2[2] - c1[2])) )
        resWeight = np.float32(w1 + mu * (w2 - w1)) / np.float32(255)

        return Vertex(resPosition, resColor, (np.float32(255),))

    def isInBoxAA(self, minCorner, maxCorner, pos):
        if (pos[0] < minCorner[0] or pos[0] > maxCorner[0]):
            return False
        if (pos[1] < minCorner[1] or pos[1] > maxCorner[1]):
            return False
        if (pos[2] < minCorner[2] or pos[2] > maxCorner[2]):
            return False

        return True

    def extractIsoSurfaceAtPosition(self, worldPos, hashData, SDFIndex, minCorner, maxCorner):
        if (m_boxEnabled == 1):
            if not self.isInBoxAA(minCorner,maxCorner, worldPos):
                #print("not inside box")
                return


        #print("lets continue")
        isolevel = np.float32(0.0)

        P = virtualVoxelSize / np.float32(2.0)
        M = -P

        p000 = worldPos + np.array((M, M, M), dtype=np.float32)
        #print("p000 " + str(p000))
        dist000, color000, valid000 = self.trilinearInterpolationSimpleFastFast(p000, hashData, SDFIndex)
        if not valid000:
            #print("returned valid000 "+ str(worldPos))
            return

        p100 = worldPos + np.array((P, M, M), dtype=np.float32)
        #print("p100 " + str(p100))
        dist100, color100, valid100 = self.trilinearInterpolationSimpleFastFast(p100, hashData, SDFIndex)
        if not valid100:
            #print("returned valid100 "+ str(worldPos))
            return

        p010 = worldPos + np.array((M, P, M), dtype=np.float32)
        #print("p010 " + str(p010))
        dist010, color010, valid010 = self.trilinearInterpolationSimpleFastFast(p010, hashData, SDFIndex)
        if not valid010:
            #print("returned valid010 "+ str(worldPos))
            return

        p001 = worldPos + np.array((M, M, P), dtype=np.float32)
        #print("p001 " + str(p001))
        dist001, color001, valid001 = self.trilinearInterpolationSimpleFastFast(p001, hashData, SDFIndex)
        if not valid001:
            #print("returned valid001 "+ str(worldPos))
            return

        p110 = worldPos + np.array((P, P, M), dtype=np.float32)
        #print("p110 " + str(p110))
        dist110, color110, valid110 = self.trilinearInterpolationSimpleFastFast(p110, hashData, SDFIndex)
        if not valid110:
            #print("returned valid110 "+ str(worldPos))
            return

        p011 = worldPos + np.array((M, P, P), dtype=np.float32)
        #print("p011 " + str(p011))
        dist011, color011, valid011 = self.trilinearInterpolationSimpleFastFast(p011, hashData, SDFIndex)
        if not valid011:
            #print("returned valid011 "+ str(worldPos))
            return

        p101 = worldPos + np.array((P, M, P), dtype=np.float32)
        #print("p101 " + str(p101))
        dist101, color101, valid101 = self.trilinearInterpolationSimpleFastFast(p101, hashData, SDFIndex)
        if not valid101:
            #print("returned valid101 "+ str(worldPos))
            return

        p111 = worldPos + np.array((P, P, P), dtype=np.float32)
        #print("p111 " + str(p111))
        dist111, color111, valid111 = self.trilinearInterpolationSimpleFastFast(p111, hashData, SDFIndex)
        if not valid111:
            #print("returned valid111 "+ str(worldPos))
            return

        cubeindex = 0
        if (dist010 < isolevel): cubeindex += 1
        if (dist110 < isolevel): cubeindex += 2
        if (dist100 < isolevel): cubeindex += 4
        if (dist000 < isolevel): cubeindex += 8
        if (dist011 < isolevel): cubeindex += 16
        if (dist111 < isolevel): cubeindex += 32
        if (dist101 < isolevel): cubeindex += 64
        if (dist001 < isolevel): cubeindex += 128

        if (edgeTable[cubeindex] == 0 or edgeTable[cubeindex] == 255):
            #print("returned edgetablecubeindex: "+str(edgeTable[cubeindex])+ " "+str(worldPos))
            #print("dist000: "+ str(dist000))
            #print("dist001: " + str(dist001))
            #print("dist010: " + str(dist010))
            #print("dist011: " + str(dist011))
            #print("dist100: " + str(dist100))
            #print("dist101: " + str(dist101))
            #print("dist110: " + str(dist110))
            #print("dist111: " + str(dist111))
            return


        #print("we made it 2")
        distArray = np.array([dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111], dtype=np.float32)
        for k in range(0, 8):
            if abs(distArray[k]) > threshMarchingCubes2:
                #print("cant past threshold:" + str(abs(distArray[k])))
                return
            for l in range(0, 8):
                if distArray[k] * distArray[l] <  np.float32(0.0):
                    if abs(distArray[k]) + abs(distArray[l]) > threshMarchingCubes:
                        #print("abs(distArray["+str(k)+"]) + abs(distArray["+str(l)+"]) > threshMarchingCubes " + str(worldPos))
                        return
                    else:
                        if abs(distArray[k]) - abs(distArray[l]) > threshMarchingCubes:
                            #print("abs(distArray["+str(k)+"]) - abs(distArray["+str(l)+"]) > threshMarchingCubes" + str(worldPos))
                            return

        #print("we made it 3")
        v = hashData.getVoxel(worldPos, SDFIndex)
        #print("v.color: "+str(v.color))
        vertlist = [Vertex() for _ in range(12)]
        if (edgeTable[cubeindex] & 1):
            vertlist[0] = self.vertexInterp(isolevel, p010, p110, dist010, dist110, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 2):
            vertlist[1] = self.vertexInterp(isolevel, p110, p100, dist110, dist100, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 4):
            vertlist[2] = self.vertexInterp(isolevel, p100, p000, dist100, dist000, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 8):
            vertlist[3] = self.vertexInterp(isolevel, p000, p010, dist000, dist010, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 16):
            vertlist[4] = self.vertexInterp(isolevel, p011, p111, dist011, dist111, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 32):
            vertlist[5] = self.vertexInterp(isolevel, p111, p101, dist111, dist101, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 64):
            vertlist[6] = self.vertexInterp(isolevel, p101, p001, dist101, dist001, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 128):
            vertlist[7] = self.vertexInterp(isolevel, p001, p011, dist001, dist011, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 256):
            vertlist[8] = self.vertexInterp(isolevel, p010, p011, dist010, dist011, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 512):
            vertlist[9] = self.vertexInterp(isolevel, p110, p111, dist110, dist111, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 1024):
            vertlist[10] = self.vertexInterp(isolevel, p100, p101, dist100, dist101, v.color, v.color, v.weight, v.weight)
        if (edgeTable[cubeindex] & 2048):
            vertlist[11] = self.vertexInterp(isolevel, p000, p001, dist000, dist001, v.color, v.color, v.weight, v.weight)


        i = 0
        #print("we made it4")
        while triTable[cubeindex][i] != -1:
            #print("TRIANGLE TIME")
            t = Triangle(vertlist[triTable[cubeindex][i + 0]], vertlist[triTable[cubeindex][i + 1]],
                         vertlist[triTable[cubeindex][i + 2]])
            i += 3
            self.triangles.append(t)

    def linearizeChunkPos(self, chunkPos):
        p = chunkPos - self.minGridPos
        #print("p"+str(p))
        #print("gridDimensions:"+str(self.gridDimensions))
        return p[2] * self.gridDimensions[0] * self.gridDimensions[1] + p[1] * self.gridDimensions[0] + p[0]

    def delinearizeChunkIndex(self,idx):
        x = np.uintc(idx % self.gridDimensions[0])
        y = np.uintc((idx % (self.gridDimensions[0]*self.gridDimensions[1])) /self.gridDimensions[0])
        z = np.uintc(idx /( self.gridDimensions[0]*self.gridDimensions[1]))
        return self.minGridPos+ np.array((x,y,z),dtype=np.uintc)

    def extractIsoSurfaceCPU(self,minCorner,maxCorner, hashData, size):
        m_numOccupiedSDFBlocks = size
        m_SDFBlockSize = 8
        for sdfBlockId in range(0, m_numOccupiedSDFBlocks):
            #print("Block: " + str(sdfBlockId))
            for x in range(0, m_SDFBlockSize):
                for y in range(0, m_SDFBlockSize):
                    for z in range(0, m_SDFBlockSize):
                        entry = hashData.entries[sdfBlockId]
                        #print("Block: "+ str(sdfBlockId)+ " Index: "+str(x) +" "+str(y)+" "+str(z))
                        if entry.ptr != -2:
                            #print()
                            pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos)
                            #print("pi_base: " + str(pi_base))
                            pi = pi_base + np.array((x, y, z), dtype=int)
                            #print("pi: " + str(pi))
                            worldPos = hashData.virtualVoxelPosToWorld(pi)
                            #print("worldPos: " + str(worldPos))
                            self.extractIsoSurfaceAtPosition(worldPos, hashData, sdfBlockId, minCorner, maxCorner)

    def findMinVoxelPos(self, voxel_weight_thresh):
        voxel_list = list()
        m_SDFBlockSize=8
        voxelPosMin = np.array((50000 , 50000, 50000),dtype=int)
        voxelPosMax = np.array((-50000, -50000, -50000),dtype=int)
        voxel_count = 0
        voxel_weight_total = 0
        for index in range(0,self.numOccupiedChunks):
            if(self.indexing_type == 'chunkGrid'):
                print("fix here")
            else:
                hashData = self.m_grid[index]
            m_numOccupiedSDFBlocks = len(hashData.entries)


            if voxel_weight_thresh == None:
                for sdfBlockId in range(0, m_numOccupiedSDFBlocks):
                    entry = hashData.entries[sdfBlockId]
                    if entry.ptr != -2:
                        pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos)
                        for i in range(0, m_SDFBlockSize):
                            for j in range(0, m_SDFBlockSize):
                                for k in range(0, m_SDFBlockSize):
                                    voxelPos = pi_base + np.array((i, j, k), dtype=int)
                                    localsdfIndex = hashData.virtualVoxelPosToLocalSDFBlockIndex(voxelPos)
                                    voxel = hashData.SDFBlocks[512 * sdfBlockId + localsdfIndex]
                                    voxel_weight_total+=voxel.weight
                                    voxel_count+=1

        if voxel_weight_thresh == None:
            voxel_weight_thresh = (voxel_weight_total/voxel_count)
        #print("threshold: " + str(voxel_weight_thresh))
        totalBlock = 0
        #print("self.numOccupiedChunks: " + str(self.numOccupiedChunks))
        for index in range(0,self.numOccupiedChunks):
            if(self.indexing_type == 'chunkGrid'):
                print("fix here")
            else:
                hashData = self.m_grid[index]
            m_numOccupiedSDFBlocks = len(hashData.entries)

            for sdfBlockId in range(0, m_numOccupiedSDFBlocks):
                entry = hashData.entries[sdfBlockId]
                if entry.ptr != -2:
                    totalBlock+=1
                    #print("sdfBlockId: "+str(sdfBlockId)+" entry.pos: "+str(entry.pos))
                    pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos)
                    for i in range(0, m_SDFBlockSize):
                        for j in range(0, m_SDFBlockSize):
                            for k in range(0, m_SDFBlockSize):
                                voxelPos = pi_base + np.array((i, j, k), dtype=int)
                                localsdfIndex = hashData.virtualVoxelPosToLocalSDFBlockIndex(voxelPos)
                                voxel = hashData.SDFBlocks[512 * sdfBlockId + localsdfIndex]
                                if voxel.weight < voxel_weight_thresh or voxel.sdf == 0 or abs(voxel.sdf) > trunc*self.voxelSize:
                                    continue
                                voxel.pos = voxelPos
                                #print("sdfBlockId: " + str(sdfBlockId) + " entry.pos: " + str(entry.pos)+" voxelPos: "+str(voxelPos))
                                voxel_list.append(voxel)
                                for t in range(0,3):
                                    if voxelPos[t] < voxelPosMin[t]:
                                        voxelPosMin[t] = voxelPos[t]
                                    if voxelPos[t] > voxelPosMax[t]:
                                        voxelPosMax[t] = voxelPos[t]
        #print("totalBlock: "+str(totalBlock))
        return voxel_list, voxelPosMin, voxelPosMax

    def findMinMaxChunk(self,res):
        minCorner =np.array(('inf','inf','inf'),dtype=np.float32)
        maxCorner = np.array(('-inf', '-inf', '-inf'), dtype=np.float32)
        for index in range(0,self.numOccupiedChunks):
            #print("chunk: "+str(index))
            hashData = self.m_grid[index]
            chunkPos = self.gridIndexList[index]
            chunkCenter = self.chunkToWorld(chunkPos)
            chunkMinCorner = chunkCenter - (np.float32(self.voxelExtents) / np.float32(2.0)) \
                             - np.array((res, res, res), dtype=np.float32) \
                             * np.float32(HashParam_SDFBlockSize)
            chunkMaxCorner = chunkCenter + (np.float32(self.voxelExtents) / np.float32(2.0)) \
                             + np.array((res, res, res), dtype=np.float32) \
                             * np.float32(HashParam_SDFBlockSize)
            for i in range(0,3):
                if(chunkMinCorner[i]<minCorner[i]):
                    minCorner[i]=chunkMinCorner[i]
                if(chunkMaxCorner[i]>minCorner[i]):
                    maxCorner[i]=chunkMaxCorner[i]
        return minCorner,maxCorner

    def to_vox(self, dim_thresh, voxel_weight_thresh=None):
        voxel_list, voxelPosMin, voxelPosMax = self.findMinVoxelPos(voxel_weight_thresh)
        print("voxelPosMin: "+str(voxelPosMin))
        print("voxelPosMax: " + str(voxelPosMax))
        vox = Vox.Vox()
        vox.res = self.voxelSize
        vox.grid2world = np.eye(4)
        vox.grid2world[0:3, 0:3] *= np.float32(vox.res)
        vox.grid2world[2][2] = -vox.grid2world[2][2]
        vox.grid2world = np.float32(vox.grid2world)
        vox.grid2world[0][3] = voxelPosMin[0] * vox.res
        vox.grid2world[1][3] = voxelPosMin[2] * vox.res
        vox.grid2world[2][3] = -voxelPosMin[1] * vox.res
        vox.dims = np.array((voxelPosMax-voxelPosMin+np.ones(3))).astype(int)

        #TODO CHECK PADDING SHOUd BE vox.dims[0] % 16 != 1
        pad = np.array(( (16-(vox.dims[0]%16))*(vox.dims[0] % 16 != 0), (16-(vox.dims[1]%16))*(vox.dims[1] % 16 != 0), (16- (vox.dims[2]%16)) * (vox.dims[2] % 16 != 0)), dtype=int)
        vox.dims = vox.dims + pad
        if dim_thresh is not None:
            for dim in vox.dims:
                if dim_thresh< dim:
                    print("couldnt pass threshold vox.dims: "+str(vox.dims) + "\n")
                    return None
        y, z = vox.dims[1], vox.dims[2]
        vox.dims[1], vox.dims[2] = z,y
        vox.sdf = [np.float32('-0.15')] * vox.dims[0]*vox.dims[1]*vox.dims[2]
        for voxel in voxel_list:
            voxsdfIndex = self.linearizeVoxelPosVox(voxel.pos, voxelPosMin, vox.dims)
            vox.sdf[voxsdfIndex] = voxel.sdf
        vox.sdf = np.asarray(vox.sdf, dtype=np.float32).reshape([1,   vox.dims[2], vox.dims[1], vox.dims[0]])

        return vox


    def linearizeVoxelPosVox(self, virtualVoxelPos, voxelPosMin, dims):
        vox_voxelPos = virtualVoxelPos - voxelPosMin
        y = vox_voxelPos[1]
        z = vox_voxelPos[2]
        vox_voxelPos[2] = y
        vox_voxelPos[1] = z
        return vox_voxelPos[2] * dims[0] * dims[1] + vox_voxelPos[1] * dims[0] + vox_voxelPos[0]


class CUDAMarchingCubesHashSDF:
    def __init__(self):
        self.m_meshData = MeshData()
        #self.m_data

    def copyTrianglesToCPU(self, triangles):
        nTriangles = len(triangles)
        if nTriangles != 0:
            md = MeshData()
            for i in range(0, nTriangles):
                md.m_Vertices.append(triangles[i].v0.p)
                md.m_Colors.append(triangles[i].v0.c + triangles[i].v0.w)
                md.m_Vertices.append(triangles[i].v1.p)
                md.m_Colors.append(triangles[i].v1.c + triangles[i].v1.w)
                md.m_Vertices.append(triangles[i].v2.p)
                md.m_Colors.append(triangles[i].v2.c + triangles[i].v2.w)

            for i in range(0, np.uintc(len(md.m_Vertices) / 3)):
                md.m_FaceIndicesVertices.append(([3 * i, 3 * i + 1, 3 * i + 2],))

            md.mergeCloseVertices(np.float32(0.0001), True)
            self.m_meshData = md

    def toPLYfileformat(self):
        vertices = list()
        for i in range(0, len(self.m_meshData.m_Vertices)):
            v = self.m_meshData.m_Vertices[i] + self.m_meshData.m_Colors[i]
            vertices.append(v)
        vertices = np.asarray(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                                               ('blue', 'u1'), ('alpha', 'u1')])
        faces = np.asarray(self.m_meshData.m_FaceIndicesVertices, dtype=[('vertex_indices', 'i4', (3,)), ])
        return vertices, faces

class MeshData:
    def __init__(self):
        self.m_Vertices = list()
        self.m_Colors = list()
        self.m_FaceIndicesVertices = list()

    def mergeCloseVertices(self, thresh, approx):
        numV = len(self.m_Vertices)
        vertexLookUp = list()
        new_verts = list()
        new_color = list()

        cnt = np.uintc(0)
        if approx:
            neighborQuery = dict()
            for v in range(0, numV):
                vert = self.m_Vertices[v]
                coord = self.toVirtualVoxelPos(vert, thresh)
                nn = self.hasNearestNeighborApprox(coord, neighborQuery, thresh)
                if nn == -1:
                    neighborQuery[coord.tobytes()] = cnt
                    new_verts.append(vert)
                    vertexLookUp.append(cnt)
                    cnt += 1
                    new_color.append(self.m_Colors[v])
                else:
                    vertexLookUp.append(nn)

        for i in range(0, np.uintc(numV / 3)):
            self.m_FaceIndicesVertices[i] = ([vertexLookUp[3 * i], vertexLookUp[3 * i + 1], vertexLookUp[3 * i + 2]],)
        self.m_Vertices = new_verts
        self.m_Colors = new_color

    def toVirtualVoxelPos(self, pos, voxelSize):
        # TODO Change virtualvoxelsize
        p = pos / voxelSize
        return (p + np.array([np.sign(p[0]), np.sign(p[1]), np.sign(p[2])], dtype=np.float32) * np.float32(0.5)).astype(
            int)

    def hasNearestNeighborApprox(self, coord, neighborQuery, thresh):
        threshSq = np.float32(thresh * thresh)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    c = coord + np.array((i, j, k), dtype=int)
                    c_bytes = c.tobytes()
                    if c_bytes in neighborQuery:
                        return neighborQuery[c_bytes]

        return -1

def mergeCloseVertices(m_Vertices, thresh, approx):
    numV = len(m_Vertices)
    vertexLookUp = list()
    new_verts = list()
    cnt = np.uintc(0)
    if approx:
        neighborQuery = dict()
        for v in range(0, numV):
            vert = m_Vertices[v]
            coord = toVirtualVoxelPos(vert, thresh)
            nn = hasNearestNeighborApprox(coord, neighborQuery, thresh)
            if nn == -1:
                neighborQuery[coord.tobytes()] = cnt
                new_verts.append(vert)
                vertexLookUp.append(cnt)
                cnt += 1
            else:
                vertexLookUp.append(nn)

    return new_verts

def hasNearestNeighborApprox(coord, neighborQuery, thresh):
    threshSq = np.float32(thresh * thresh)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                c = coord + np.array((i, j, k), dtype=int)
                c_bytes = c.tobytes()
                if c_bytes in neighborQuery:
                    return neighborQuery[c_bytes]

    return -1

def toVirtualVoxelPos(pos, voxelSize):
    # TODO Change virtualvoxelsize
    p = pos / voxelSize
    return (p + np.array([np.sign(p[0]), np.sign(p[1]), np.sign(p[2])], dtype=np.float32) * np.float32(0.5)).astype(
        int)

def frac(a):
    # b = np.array([0,0,0], dtype=np.float32)
    # for i in range(0,3):
    #     if a[i]>0:
    #         b[i] = a[i]-math.floor(a[i])
    #     else:
    #         b[i] = a[i]-math.ceil(a[i])
    # return b
    return np.array((a[0]-math.floor(a[0]), a[1]-math.floor(a[1]), a[2]-math.floor(a[2])), dtype=np.float32)
