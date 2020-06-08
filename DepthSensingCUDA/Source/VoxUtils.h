#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <torch/torch.h>

struct Vox {
	Eigen::Vector3i dims;
	float res;
	Eigen::Matrix4f grid2world;
	torch::Tensor sdf;

	Eigen::Vector3f voxel2World(int i, int j, int k) {
		return (grid2world*Eigen::Vector4f(i, j, k, 1.0f)).topRows(3);
	}

	
};