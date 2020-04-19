#pragma once
#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include <Eigen.h>

struct Vox {
	Vector3i dims;
	Matrix4f grid2world; 
	float res;
	torch::Tensor sdf; // 1 x 1 x dims[0] x dims[1] x dims[2] dims

};

inline Vox load_vox(std::string filename, bool is_cad = false) {

	std::ifstream f(filename, std::ios::binary);
	assert(f.is_open());

	Vox vox;
	std::vector<float> sdf;
	f.read((char*)vox.dims.data(), 3 * sizeof(int));
	f.read((char*)&vox.res, sizeof(float));
	f.read((char*)vox.grid2world.data(), 16 * sizeof(float));
	int n_elems = vox.dims[0] * vox.dims[1] * vox.dims[2];

	sdf.resize(n_elems);
	f.read((char*)sdf.data(), n_elems * sizeof(float));
	if (is_cad) {
		vox.sdf = torch::from_blob(sdf.data(), { 1, vox.dims[2],vox.dims[1],vox.dims[0] }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
	}
	else {
		vox.sdf = torch::from_blob(sdf.data(), { 1, 1, vox.dims[2],vox.dims[1],vox.dims[0] }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
	}

	f.close();

	return vox;


}

inline Vox makeVoxFromSceneRepHashSDF(int* min_pos, int* dims, float& res, float* sdf) {
	Vox v;
	v.res = res;
	v.grid2world = Matrix4f::Identity();
	v.grid2world(0,0)= res, v.grid2world(1,1)= res, v.grid2world(2,2) = -res;
	v.grid2world(0,3) = min_pos[0] * res;
	v.grid2world(1,3) = min_pos[1] * res;
	v.grid2world(2,3) = -min_pos[2] * res;
	v.dims(0) = dims[0], v.dims(1) = dims[1], v.dims(2) = dims[2];
	int n_elems = dims[0] * dims[1] * dims[2];
	
	v.sdf = torch::from_blob(sdf, { 1, 1,dims[2],dims[1],dims[0] }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
	return v;
}