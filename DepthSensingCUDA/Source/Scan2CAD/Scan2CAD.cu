
#include <cutil_inline.h>
#include <cutil_math.h>

#include "cuda_SimpleMatrixUtil.h"

#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"


#define T_PER_BLOCK 8



//Torch Pass 1: Find minimum and maximum vox grid positions in the frame
//TODO: Make this with reduction
__global__ void findMinMaxVoxGridPosKernel(HashData hashData, int3* d_voxDims) {


	const HashEntry& entry = hashData.d_hashCompactified[blockIdx.x];
	uint i = threadIdx.x;	//inside of an SDF block
	Voxel v = hashData.d_SDFBlocks[entry.ptr + i];
	if (v.weight > 20 && v.sdf != float(0) && abs(v.sdf) <= 0.03f) {
		int3 pos = hashData.SDFBlockToVirtualVoxelPos(entry.pos) + make_int3(hashData.delinearizeVoxelIndex(i));
		//printf("min.pos: %d,%d,%d entry.pos: %d,%d,%d hashId: %d\n", minPos.x, minPos.y, minPos.z, entry.pos.x, entry.pos.y, entry.pos.z,  hashIdx);
		//printf("pos %d %d %d\n", pos.x, pos.z, pos.y);
		int3& minPos = d_voxDims[0];
		int3& maxPos = d_voxDims[1];
		if (pos.x < minPos.x) {
			minPos.x = pos.x;
		}
		if (pos.y < minPos.y) {
			minPos.y = pos.y;
		}
		if (pos.z < minPos.z) {
			minPos.z = pos.z;
		}

		if (pos.x > maxPos.x) {
			maxPos.x = pos.x;
		}
		if (pos.y > maxPos.y) {
			maxPos.y = pos.y;
		}
		if (pos.z > maxPos.z) {
			maxPos.z = pos.z;
		}


	}

}

//Torch Pass 2: Fill tensor
__global__ void fillTensorKernel(float* d_sdf, int n_elems) {

	uint i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < n_elems) {
		d_sdf[i] = -0.15;
	}


}

//Torch Pass 3: Create SDF tensor according to Scan2CAD index
__global__ void createSDFTensorKernel(HashData hashData, float* d_sdf, int3 minPos, int3 dims) {

	const HashEntry& entry = hashData.d_hashCompactified[blockIdx.x];
	uint i = threadIdx.x;	//inside of an SDF block
	uint idx = entry.ptr + i;
	Voxel v = hashData.d_SDFBlocks[idx];
	int3 pi = hashData.SDFBlockToVirtualVoxelPos(entry.pos);

	int index = -1;
	if (v.weight > 20 && v.sdf != float(0) && abs(v.sdf) <= 0.03f) {
		int3 voxelPos = pi + make_int3(hashData.delinearizeVoxelIndex(i)) - minPos;

		int y = voxelPos.y, z = voxelPos.z;
		voxelPos.y = z, voxelPos.z = y;
		index = voxelPos.z * dims.y * dims.x + voxelPos.y * dims.x + voxelPos.x;

		d_sdf[index] = v.sdf;

	}


}

extern "C" float* createSDFTensor(HashData & hashData, const HashParams & hashParams, int* min_pos, int* h_dims) {
	float* h_sdf;
	int3 h_voxDims[2];
	//h_voxDims = new  int3[2];
	h_voxDims[0] = make_int3(200);//min initilization
	h_voxDims[1] = make_int3(0); //max initilization
	int3* d_voxDims;

	cutilSafeCall(cudaMalloc(&d_voxDims, sizeof(int3) * 2));
	cutilSafeCall(cudaMemcpy(d_voxDims, &h_voxDims, sizeof(int3) * 2, cudaMemcpyHostToDevice));
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	if (hashParams.m_numOccupiedBlocks > 0) {
		findMinMaxVoxGridPosKernel << <gridSize, blockSize >> > (hashData, d_voxDims);
		int3 voxDims[2];

		cutilSafeCall(cudaMemcpy(&voxDims, d_voxDims, sizeof(int3) * 2, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaFree(d_voxDims));
		// z and y is different in Scan2CAD
		if (voxDims[1].x > voxDims[0].x && voxDims[1].y > voxDims[0].y && voxDims[1].z > voxDims[0].z) {
			min_pos[0] = voxDims[0].x, min_pos[1] = voxDims[0].z, min_pos[2] = voxDims[0].y;
			int3 dims = voxDims[1] - voxDims[0] + make_int3(1);
			int y = dims.y, z = dims.z;
			dims.y = z, dims.z = y;
			int3 pad = make_int3((16 - (dims.x % 16)) * (dims.x % 16 != 0), (16 - (dims.y % 16)) * (dims.y % 16 != 0), (16 - (dims.z % 16)) * (dims.z % 16 != 0));
			dims = dims + pad;

			//std::cout << "max " << voxDims[1].x << " " << voxDims[1].z << " " << voxDims[1].y << std::endl;
			//std::cout << "min " << voxDims[0].x << " " << voxDims[0].z << " " << voxDims[0].y << std::endl;
			//std::cout << "dims " << dims.x << " " << dims.y << " " << dims.z << std::endl;
			h_dims[0] = dims.x, h_dims[1] = dims.y, h_dims[2] = dims.z;
			int n_elems = dims.x * dims.y * dims.z;

			h_sdf = new float[n_elems];
			for (int i = 0; i < n_elems; i++) {
				h_sdf[i] = -0.15;
			}
			if (n_elems > 1) {
				float* d_sdf;
				cutilSafeCall(cudaMalloc(&d_sdf, sizeof(float) * n_elems));

				const dim3 gridSize2((n_elems + (T_PER_BLOCK * T_PER_BLOCK) - 1) / (T_PER_BLOCK * T_PER_BLOCK), 1);
				const dim3 blockSize2((T_PER_BLOCK * T_PER_BLOCK), 1);

				fillTensorKernel << <gridSize2, blockSize2 >> > (d_sdf, n_elems);
				createSDFTensorKernel << <gridSize, blockSize >> > (hashData, d_sdf, voxDims[0], dims);
				cutilSafeCall(cudaMemcpy(h_sdf, d_sdf, sizeof(float) * n_elems, cudaMemcpyDeviceToHost));
				cutilSafeCall(cudaFree(d_sdf));


			}
		}
		else {
			h_dims[0] = 0, h_dims[1] = 0, h_dims[2] = 0;
		}
	}
	return h_sdf;
}