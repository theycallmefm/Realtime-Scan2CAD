//#include <cutil_inline.h>
//#include <cutil_math.h>
//
//#include "cuda_SimpleMatrixUtil.h"
//
//#include "VoxelUtilHashSDF.h"
//#include "DepthCameraUtil.h"
//
//#define T_PER_BLOCK 8
//
////Torch Pass 1: Find minimum and maximum vox grid positions in the frame
////TODO: Make this with reduction
//__global__ void findMinMaxVoxGridPosKernel(HashData hashData) {
//
//	//const uint hashIdx = blockIdx.x;
//
//	const uint hashIdx = blockIdx.x*blockDim.x + threadIdx.x;
//
//
//
//	if (hashIdx < c_hashParams.m_numOccupiedBlocks) {
//
//
//		const HashEntry& entry = hashData.d_hashCompactified[hashIdx];
//
//
//
//		const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
//
//#pragma unroll 1
//		for (uint i = 0; i < linBlockSize; i++) {	//clear sdf block: CHECK TODO another kernel?
//			Voxel v = hashData.d_SDFBlocks[entry.ptr + i];
//			if (v.weight > 20) {
//				int3 pos = hashData.SDFBlockToVirtualVoxelPos(entry.pos) + make_int3(hashData.delinearizeVoxelIndex(i));
//				//printf("min.pos: %d,%d,%d entry.pos: %d,%d,%d hashId: %d\n", minPos.x, minPos.y, minPos.z, entry.pos.x, entry.pos.y, entry.pos.z,  hashIdx);
//				int3 & minPos = hashData.d_voxDims[0];
//				int3 & maxPos = hashData.d_voxDims[1];
//				if (pos.x < minPos.x) {
//					minPos.x = pos.x;
//				}
//				if (pos.y < minPos.y) {
//					minPos.y = pos.y;
//				}
//				if (pos.z < minPos.z) {
//					minPos.z = pos.z;
//				}
//
//				if (pos.x > maxPos.x) {
//					maxPos.x = pos.x;
//				}
//				if (pos.y >	maxPos.y) {
//					maxPos.y = pos.y;
//				}
//				if (pos.z > maxPos.z) {
//					maxPos.z = pos.z;
//				}
//
//			}
//		}
//
//	}
//}
//
//
//extern "C" void findMinMaxVoxGridPos(HashData& hashData, const HashParams& hashParams) {
//
//	const unsigned int threadsPerBlock = T_PER_BLOCK * T_PER_BLOCK;
//	const dim3 gridSize((hashParams.m_numOccupiedBlocks + threadsPerBlock - 1) / threadsPerBlock, 1);
//	const dim3 blockSize(threadsPerBlock, 1);
//
//	if (hashParams.m_numOccupiedBlocks > 0) {
//		std::cout << "m_numOccupiedBlocks: " << hashParams.m_numOccupiedBlocks << std::endl;
//		findMinMaxVoxGridPosKernel << <gridSize, blockSize >> >(hashData);
//	}
//#ifdef _DEBUG
//	cutilSafeCall(cudaDeviceSynchronize());
//	cutilCheckMsg(__FUNCTION__);
//#endif
//	int3 voxDims[2];
//
//	cutilSafeCall(cudaMemcpy(&voxDims, hashData.d_voxDims, sizeof(int3) * 2, cudaMemcpyDeviceToHost));
//	std::cout << "maxPos: " << voxDims[1].x << " " << voxDims[1].y << " " << voxDims[1].z << std::endl;
//	std::cout << "minPos: " << voxDims[0].x << " " << voxDims[0].y << " " << voxDims[0].z << std::endl;
//	//voxDims[0] = make_int3(INT_MAX);
//	//voxDims[1] = make_int3(0);
//	//cutilSafeCall(cudaMemcpy(hashData.d_voxDims, voxDims, sizeof(int3) * 2, cudaMemcpyHostToDevice));
//}
//
////Torch Pass 2: Create sdf
//__global__ void createSDFTensorKernel(HashData hashData) {
//
//	//const uint hashIdx = blockIdx.x;
//
//	const uint hashIdx = blockIdx.x*blockDim.x + threadIdx.x;
//
//
//
//	if (hashIdx < c_hashParams.m_numOccupiedBlocks) {
//
//
//		const HashEntry& entry = hashData.d_hashCompactified[hashIdx];
//
//
//
//		const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
//
//#pragma unroll 1
//		for (uint i = 0; i < linBlockSize; i++) {	//clear sdf block: CHECK TODO another kernel?
//			Voxel v = hashData.d_SDFBlocks[entry.ptr + i];
//			if (v.weight > 20) {
//				int3 pos = hashData.SDFBlockToVirtualVoxelPos(entry.pos) + make_int3(hashData.delinearizeVoxelIndex(i));
//				//printf("min.pos: %d,%d,%d entry.pos: %d,%d,%d hashId: %d\n", minPos.x, minPos.y, minPos.z, entry.pos.x, entry.pos.y, entry.pos.z,  hashIdx);
//				int3 & minPos = hashData.d_voxDims[0];
//				int3 & maxPos = hashData.d_voxDims[1];
//				if (pos.x < minPos.x) {
//					minPos.x = pos.x;
//				}
//				if (pos.y < minPos.y) {
//					minPos.y = pos.y;
//				}
//				if (pos.z < minPos.z) {
//					minPos.z = pos.z;
//				}
//
//				if (pos.x > maxPos.x) {
//					maxPos.x = pos.x;
//				}
//				if (pos.y >	maxPos.y) {
//					maxPos.y = pos.y;
//				}
//				if (pos.z > maxPos.z) {
//					maxPos.z = pos.z;
//				}
//
//			}
//		}
//
//	}
//}
//
//extern "C" void createSDFTensor(HashData& hashData, const HashParams& hashParams) {
//
//	const unsigned int threadsPerBlock = T_PER_BLOCK * T_PER_BLOCK;
//	const dim3 gridSize((hashParams.m_numOccupiedBlocks + threadsPerBlock - 1) / threadsPerBlock, 1);
//	const dim3 blockSize(threadsPerBlock, 1);
//
//	if (hashParams.m_numOccupiedBlocks > 0) {
//		std::cout << "m_numOccupiedBlocks: " << hashParams.m_numOccupiedBlocks << std::endl;
//		findMinMaxVoxGridPosKernel << <gridSize, blockSize >> >(hashData);
//	}
//#ifdef _DEBUG
//	cutilSafeCall(cudaDeviceSynchronize());
//	cutilCheckMsg(__FUNCTION__);
//#endif
//	int3 voxDims[2];
//
//	cutilSafeCall(cudaMemcpy(&voxDims, hashData.d_voxDims, sizeof(int3) * 2, cudaMemcpyDeviceToHost));
//	std::cout << "maxPos: " << voxDims[1].x << " " << voxDims[1].y << " " << voxDims[1].z << std::endl;
//	std::cout << "minPos: " << voxDims[0].x << " " << voxDims[0].y << " " << voxDims[0].z << std::endl;
//	//voxDims[0] = make_int3(INT_MAX);
//	//voxDims[1] = make_int3(0);
//	//cutilSafeCall(cudaMemcpy(hashData.d_voxDims, voxDims, sizeof(int3) * 2, cudaMemcpyHostToDevice));
//}