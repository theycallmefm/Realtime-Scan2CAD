##owner: https://github.com/skanti
import sys
import os
import struct
import glob
import numpy as np
import torch
import hashgrid
import h5py

class Vox:
    def __init__(self, dims=None, res=None, grid2world=None, sdf=None, pdf=None, noc=None, bbox=None):
        self.dims = dims
        self.res = res
        self.grid2world = grid2world
        self.sdf = sdf
        self.pdf = pdf
        self.noc = noc
        self.bbox = bbox

    def make_torch(self):
        self.sdf = torch.as_tensor(self.sdf)
        self.grid2world = torch.as_tensor(self.grid2world)
        if self.pdf is not None:
            self.pdf = torch.as_tensor(self.pdf)
        if self.noc is not None:
            self.noc = torch.as_tensor(self.noc)
    
    def make_numpy(self):
        self.sdf = np.asarray(self.sdf)
        self.grid2world = np.asarray(self.grid2world)
        if self.pdf is not None:
            self.pdf = np.asarray(self.pdf)
        if self.noc is not None:
            self.noc = np.asarray(self.noc)

def write_hdf5(filename, s):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('res', (1,), data=s.res, compression='gzip')
        f.create_dataset('grid2world', (1, 4, 4), data=s.grid2world, compression='gzip')
        f.create_dataset('dims', (1, 3), data=s.dims, compression='gzip')
        f.create_dataset("sdf", (1, 1, s.dims[2], s.dims[1], s.dims[0],), data=s.sdf, compression='gzip')
        if s.pdf is not None:
            f.create_dataset("pdf", (1, 1, s.dims[2], s.dims[1], s.dims[0],), data=s.pdf, compression='gzip')
        if s.noc is not None:
            f.create_dataset("noc", (1, 3, s.dims[2], s.dims[1], s.dims[0],), data=s.noc, compression='gzip')
        if s.bbox is not None:
            f.create_dataset("bbox", (1, 1, s.dims[2], s.dims[1], s.dims[0],), data=s.bbox, compression='gzip')

def load_hdf5(filename):
    s = Vox()
    with h5py.File(filename, 'r') as f:
        for key in list(f.keys()):
            if key == 'dims':
                s.dims = f[key][(0)]
            if key == 'res':
                s.res = f[key][(0)]
            if key == 'grid2world':
                s.grid2world = f[key][(0)]
                s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order='F')
            if key == 'sdf':
                s.sdf = f[key][(0)]
            if key == 'pdf':
                s.pdf = f[key][(0)]
            if key == 'noc':
                s.noc = f[key][(0)]
            if key == 'bbox':
                s.bbox = f[key][(0)]
    print("v.grid2world"+str(s.grid2world))
    return s

def load_vox_header(filename):
    assert os.path.isfile(filename), "file not found: %s" % filename
    if filename.endswith(".df"):
        f_or_c = "C"
    elif filename.endswith(".sdf"):
        f_or_c = "C"
    else:
        f_or_c = "F"

    fin = open(filename, 'rb')
    
    s = Vox()
    s.dims = [0,0,0]
    s.dims[0] = struct.unpack('I', fin.read(4))[0]
    s.dims[1] = struct.unpack('I', fin.read(4))[0]
    s.dims[2] = struct.unpack('I', fin.read(4))[0]
    s.res = struct.unpack('f', fin.read(4))[0]
    n_elems = s.dims[0]*s.dims[1]*s.dims[2]

    s.grid2world = struct.unpack('f'*16, fin.read(16*4))
    s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order=f_or_c)

    return s


def load_vox(filename):
    assert os.path.isfile(filename), "file not found: %s" % filename
    if filename.endswith(".df"):
        f_or_c = "C"
    elif filename.endswith(".sdf"):
        f_or_c = "C"
    else:
        f_or_c = "F"

    fin = open(filename, 'rb')
    
    s = Vox()
    s.dims = [0,0,0]
    s.dims[0] = struct.unpack('I', fin.read(4))[0]
    s.dims[1] = struct.unpack('I', fin.read(4))[0]
    s.dims[2] = struct.unpack('I', fin.read(4))[0]
    s.res = struct.unpack('f', fin.read(4))[0]
    n_elems = s.dims[0]*s.dims[1]*s.dims[2]

    s.grid2world = struct.unpack('f'*16, fin.read(16*4))
    s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order=f_or_c)

    # -> sdf 1-channel
    s.sdf = struct.unpack('f'*n_elems, fin.read(n_elems*4))
    s.sdf = np.asarray(s.sdf, dtype=np.float32).reshape([1, s.dims[2], s.dims[1], s.dims[0]])
    # <-

    # -> pdf 1-channel
    pdf_bytes = fin.read(n_elems*4)
    if pdf_bytes:
        s.pdf = struct.unpack('f'*n_elems, pdf_bytes)
        s.pdf = np.asarray(s.pdf, dtype=np.float32).reshape([1, s.dims[2], s.dims[1], s.dims[0]])
    # <-

    # -> noc 3-channels
    noc_bytes = fin.read(3*n_elems*4)
    if noc_bytes:
        s.noc = struct.unpack('f'*n_elems*3, noc_bytes)
        s.noc = np.asarray(s.noc, dtype=np.float32).reshape([3, s.dims[2], s.dims[1], s.dims[0]])
    # <-
    
    # -> bbox 1-channel
    bbox_bytes = fin.read(n_elems*4)
    if bbox_bytes:
        s.bbox = struct.unpack('f'*n_elems, bbox_bytes)
        s.bbox = np.asarray(s.bbox, dtype=np.float32).reshape([1, s.dims[2], s.dims[1], s.dims[0]])
    # <-
    fin.close()

    return s

def write_vox(filename, s):
    fout = open(filename, 'wb')
    fout.write(struct.pack('I', s.dims[0]))
    fout.write(struct.pack('I', s.dims[1]))
    fout.write(struct.pack('I', s.dims[2]))
    fout.write(struct.pack('f', s.res))
    n_elems = np.prod(s.dims)
    fout.write(struct.pack('f'*16, *s.grid2world.flatten('F')))
    fout.write(struct.pack('f'*n_elems, *s.sdf.flatten('C')))
    if s.pdf is not None:
        fout.write(struct.pack('f'*n_elems, *s.pdf.flatten('C')))
    if s.noc is not None:
        fout.write(struct.pack('f'*n_elems*3, *s.noc.flatten('C')))
    if s.bbox is not None:
        fout.write(struct.pack('f'*n_elems, *s.bbox.flatten('C')))
    fout.close()

def load_scdf(filename):
    assert os.path.isfile(filename), "file not found: %s" % filename

    fin = open(filename, 'rb')
    
    s = Vox()
    s.dims = [0,0,0]
    s.dims[0] = struct.unpack('I'*2, fin.read(4*2))[0]
    s.dims[1] = struct.unpack('I'*2, fin.read(4*2))[0]
    s.dims[2] = struct.unpack('I'*2, fin.read(4*2))[0]
    s.res = struct.unpack('f', fin.read(4))[0]
    n_elems = s.dims[0]*s.dims[1]*s.dims[2]

    bbox_min = [0,0,0]
    bbox_min = struct.unpack('f'*3, fin.read(4*3))
    bbox_max = [0,0,0]
    bbox_max = struct.unpack('f'*3, fin.read(4*3))
    bbox = np.array(bbox_max) - np.array(bbox_min)
    s.grid2world = np.eye(4)
    s.grid2world[0:3,3] = np.array(bbox_min) - 16*s.res
    s.grid2world[0:3,0:3] *= s.res

    # -> sdf 1-channel
    s.sdf = struct.unpack('f'*n_elems, fin.read(n_elems*4))
    s.sdf = np.asarray(s.sdf, dtype=np.float32).reshape([1, s.dims[2], s.dims[1], s.dims[0]])*s.res
    # <-

    fin.close()

    return s

def load_hashgrid(filename, dim_thresh, voxel_weight_thresh):
    hg = hashgrid.CUDASceneRepChunkGrid(filename, "")
    return hg.to_vox(dim_thresh, voxel_weight_thresh)