import Vox
import numpy as np
import os
import torch
import JSONHelper
import multiprocessing as mp

params = JSONHelper.read("E:\Research\scan2cad-new\data\parameters.json")
class Args:
    def __init__(self):
        self.hashgrid_filename = None
        self.voxel_weight_thresh = 20
        self.out_path = "C:\scannet\scannet_hdf5"
        self.filetype = "hdf5"
        self.dim_thresh = 1000000
        self.make_rot = True
        self.scene_id ="test"


def find_jobs(hashgrid_path, out_path, scenes=None):
    jobs=[]
    for dirName, subDirList, fileList in os.walk(hashgrid_path):
        for file in fileList:
            if file[-9:] == ".hashgrid" and (scenes is None or file[:12] in scenes):
                fileName = dirName + "/" + file
                scene_id = file[:12]
                if not os.path.exists(os.path.join(out_path, scene_id)):
                    os.mkdir(os.path.join(out_path, scene_id))
                    job = Args()
                    job.hashgrid_filename = fileName
                    job.scene_id =scene_id
                    jobs.append(job)
    print(str(len(jobs))+" scenes are going to be converted")
    return jobs

def rotate_grid(rot, vox):
    angle = 0.5 * rot * np.pi
    c = np.cos(angle)
    s = np.sin(angle)
    rot_y = torch.tensor([
        [c, 0, -s, 0],
        [0, 1, 0, 0],
        [s, 0, c, 0],
        [0, 0, 0, 1]])

    dims_old = [d for d in vox.dims]
    if rot == 0:
        frot = lambda x: x
    elif rot == 1:
        frot = lambda x: x.transpose(1, 3).flip(1)
        rot_y[0, 3] = vox.dims[0] - 1.0
        vox.dims[0], vox.dims[2] = vox.dims[2], vox.dims[0]
    elif rot == 2:
        frot = lambda x: x.flip(1).flip(3)
        rot_y[0, 3] = vox.dims[0] - 1.0
        rot_y[2, 3] = vox.dims[2] - 1.0
    elif rot == 3:
        frot = lambda x: x.transpose(1, 3).flip(3)
        rot_y[2, 3] = vox.dims[2] - 1.0
        vox.dims[0], vox.dims[2] = vox.dims[2], vox.dims[0]

    # tmp = torch.mm(rot_y, torch.tensor([0, 0, 0, 1.0]).unsqueeze(1))
    # tmp = torch.mm(rot_y, torch.tensor([vox.dims[0] - 1, 0, vox.dims[2] - 1, 1.0]).unsqueeze(1))
    # vox.sdf = torch.as_tensor(vox.sdf)
    vox.sdf = frot(vox.sdf)
    if vox.pdf is not None:
        vox.pdf = frot(vox.pdf)
    if vox.noc is not None:
        vox.noc = frot(vox.noc)

    assert np.linalg.det(rot_y[:3, :3]) > 0.99  # <-- approx 1.0
    vox.grid2world = torch.mm(vox.grid2world, rot_y)
    return vox


def convert_hashgrid(args):
    fileName = args.hashgrid_filename
    scene_id = args.scene_id
    out_path = args.out_path
    filetype = args.filetype
    dim_thresh = args.dim_thresh
    voxel_weight_thresh = args.voxel_weight_thresh
    make_rot = args.make_rot
    vox = Vox.load_hashgrid(fileName, dim_thresh, voxel_weight_thresh)
    if vox is not None:
        o_path = os.path.join(out_path, scene_id, scene_id) + "_res30mm_rot0."+filetype
        if filetype == 'vox':
            Vox.write_vox(o_path, vox)
        elif filetype == 'hdf5':
            Vox.write_hdf5(o_path,vox)
        if make_rot:
            for rot in range(1, 4):
                vox.make_torch()
                new_vox = rotate_grid(rot, vox)
                new_vox.make_numpy()
                o_path = os.path.join(out_path, scene_id, scene_id) + "_res30mm_rot" + str(rot) + "."+filetype
                if filetype == 'vox':
                    Vox.write_vox(o_path, vox)
                elif filetype == 'hdf5':
                    Vox.write_hdf5(o_path, vox)
    print("converted "+fileName)

def test_hdf5noc(hdf5_path,vox_path,scene_id):
    vox_path = os.path.join(vox_path,scene_id,scene_id+"_res30mm_rot0.vox")
    hdf5_path = os.path.join(hdf5_path, scene_id, scene_id + "_res30mm_rot0.hdf5")

    v = Vox.load_hdf5(hdf5_path)
    Vox.write_vox(vox_path,v)
    print("saved to "+str(vox_path))
if __name__ == '__main__':
    hashgrid_path = params["scannet_hashgrid"]
    vox_path = params["scannet_vox"]
    #voxnoc_path = params["scannet_voxnoc"]
    hdf5_path = params["scannet_hdf5"]
    #hdf5noc_path = params["scannet_hdf5noc"]
    #scenes = ['scene0103_00']
    #vox = Vox.load_hashgrid(params["scannet_hashgrid"]+"scene0470_00/scene0470_00.hashgrid", 1000000, 20)
    test_hdf5noc(hdf5_path,vox_path,"scene0033_00")
    voxel_hashing_args =Args()

    voxel_hashing_path = "E:/Research/VoxelHashing/DepthSensingCUDA"
    voxel_hashing_args.out_path = voxel_hashing_path
    voxel_hashing_args.filetype ="vox"
    voxel_hashing_args.hashgrid_filename="E:/Research/VoxelHashing/DepthSensingCUDA/test.hashgrid"
    convert_hashgrid(voxel_hashing_args)
    #convert_hashgrid(hashgrid_path, vox_path, "vox", scenes)
    #test_hdf5noc(hashgrid_path, hdf5_path, vox_path, hdf5noc_path,voxnoc_path,scenes[0])
    #jobs =find_jobs(hashgrid_path, hdf5_path)
    #pool = mp.Pool(processes=3)
    #pool.map(convert_hashgrid, jobs)
    #convert_hashgrid(jobs[0])
