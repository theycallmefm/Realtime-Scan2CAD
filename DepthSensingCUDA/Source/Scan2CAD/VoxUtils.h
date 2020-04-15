#pragma once
#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <torch/torch.h>
//#include <cutil_math.h>

struct Vox {
	torch::Tensor dims; // 3 dims
	float res =0.03;
	torch::Tensor grid2world; // 4x4 dims
	torch::Tensor sdf; // 1 x 1 x dims[0] x dims[1] x dims[2] dims

};

inline Vox load_vox(std::string filename, bool is_cad = false) {

	std::ifstream f(filename, std::ios::binary);
	assert(f.is_open());

	Vox vox;
	std::vector<int32_t> dims;
	float res;
	std::vector<float> grid2world;
	std::vector<float> sdf;
	dims.resize(3);
	grid2world.resize(16);
	f.read((char*)dims.data(), 3 * sizeof(int32_t));
	f.read((char*)&res, sizeof(float));
	f.read((char*)grid2world.data(), 16 * sizeof(float));
	int n_elems = dims[0] * dims[1] * dims[2];

	sdf.resize(n_elems);
	f.read((char*)sdf.data(), n_elems * sizeof(float));

	vox.dims = torch::from_blob(dims.data(), { 3 }, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

	vox.grid2world = torch::from_blob(grid2world.data(), { 4,4 }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
	vox.res = res;
	if (is_cad) {
		vox.sdf = torch::from_blob(sdf.data(), { 1, dims[2],dims[1],dims[0] }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
	}
	else {
		vox.sdf = torch::from_blob(sdf.data(), { 1, 1, dims[2],dims[1],dims[0] }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
		std::cout << "vox.dims: " << vox.dims << std::endl;
		std::cout << "vox.grid2world: " << vox.grid2world << std::endl;
	}

	f.close();

	return vox;


}

inline Vox makeVoxFromSceneRepHashSDF(int* min_pos, int* dims, float& res, float* sdf) {
	Vox v;
	v.res = res;
	v.grid2world = torch::eye(4).to(at::Device(torch::kCUDA));
	v.grid2world[0][0], v.grid2world[1][1], v.grid2world[2][2] = res;
	v.grid2world[0][3] = min_pos[0] * res;
	v.grid2world[1][3] = min_pos[1] * res;
	v.grid2world[2][3] = -min_pos[2]* res;
	
	v.dims = torch::ones(3);
	v.dims[0] = dims[0], v.dims[1] = dims[1], v.dims[2] = dims[2];
	int n_elems = dims[0] * dims[1] * dims[2];
	
	v.sdf = torch::from_blob(sdf, { 1, 1,dims[2],dims[1],dims[0] }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
	return v;
}