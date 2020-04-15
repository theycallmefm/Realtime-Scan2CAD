#pragma once
#undef V
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <random>
#include <fstream>
#include<tuple>
#include "VoxUtils.h"
#include "hungarian-algorithm-cpp/Hungarian.h"

//extern "C" void findMinMaxVoxGridPos(HashData & hashData, const HashParams & hashParams);

class Scan2CAD
{
public:
	Scan2CAD() {
		create();
	}

	~Scan2CAD(void) {
		destroy();
	}

	
	std::unordered_map<std::string, at::Tensor> forward(Vox& v);


private:

	void create();
	void destroy(void);

	void loadLatentSpace(const std::string folder);
	void loadCADsdf(const std::string folder);
	void loadModules(const std::string folder);
	void loadTestPool(const std::string folder);
	
	torch::Tensor nms(int kernel_size, at::Tensor x);
	void calcCenteredCrops(at::Tensor& center, at::Tensor& xdims, at::Tensor& dims, at::Tensor& smin, at::Tensor& smax, at::Tensor& tmin, at::Tensor& tmax);
	void cropCenterCopy(at::Tensor& smin, at::Tensor& smax, at::Tensor& src, at::Tensor& tmin, at::Tensor& tmax, at::Tensor& target);
	torch::Tensor makeCoord(at::Tensor dims);
	
	void retrievalByOptimalAssignment(at::Tensor& z_queries, std::vector<unsigned int>& survived, std::vector<std::string>& cadkey);
	void calculateRotationViaProcrustes(at::Tensor& noc, at::Tensor& mask, at::Tensor& scale, at::Tensor& factor_interpolate, at::Tensor& grid2world, at::Tensor& rots);
	
	
	torch::Tensor feedForwardObjectDetection(Vox& v, at::Tensor& ft_pred);
	std::unordered_map<std::string, at::Tensor> feedForwardObject(Vox& v, torch::Tensor& ft_pred, torch::Tensor& ft_crop);
	

	torch::jit::script::Module backbone, decode, feature2heatmap0, feature2descriptor, block0, feature2mask, feature2noc, feature2scale;
	std::unordered_map<std::string, at::Tensor> cadkey2latent, cadkey2sdf;
	const std::string checkpoint_path;
	std::vector<std::string> cadkey_pool;


	int batch_size = 1;
	double thresh_objectness = 0.5;
	int canonical_cube = 32;

	std::vector<std::unordered_map<std::string, at::Tensor>> collector;

};