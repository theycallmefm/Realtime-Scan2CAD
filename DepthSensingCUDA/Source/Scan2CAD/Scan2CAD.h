#pragma once
//TODO: there is a macro V in DXUT which messes with torch library, check if undef is optimal solution
#undef V
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <random>
#include <fstream>
#include<tuple>
#include "VoxUtils.h"
#include "VoxelUtilHashSDF.h"
#include "hungarian-algorithm-cpp/Hungarian.h"
#include <Eigen.h>
#include "GlobalScan2CADState.h"
#include "../GlobalAppState.h"
#include "CAD.h"

extern "C" float* createSDFTensor(HashData & hashData, const HashParams & hashParams, int* min_pos, int* dims);

class Scan2CAD
{
public:
	Scan2CAD() {
		create();
	}

	~Scan2CAD(void) {
		destroy();
	}

	
	//std::unordered_map<std::string, Matrix4f> forward(Vox& v);
	std::vector<CAD> forward(HashData& hashData, const HashParams& hashParams);


private:

	void create();
	void destroy(void);

	void loadLatentSpaceAll();
	void loadLatentSpace();
	void loadCADsdfAll();
	at::Tensor loadCADsdf(std::vector<std::string>& cadkeys);
	void loadModules();
	void loadTestPool();
	
	//torch::Tensor nms(int kernel_size, at::Tensor x);
	void nms(int kernel_size, at::Tensor& x);
	void calcCenteredCrops(at::Tensor& center, at::Tensor& xdims, at::Tensor& dims, at::Tensor& smin, at::Tensor& smax, at::Tensor& tmin, at::Tensor& tmax);
	void cropCenterCopy(at::Tensor& smin, at::Tensor& smax, at::Tensor& src, at::Tensor& tmin, at::Tensor& tmax, at::Tensor& target);
	void calcCenteredCropsAndCropCenterCopy(std::array <int, 3>& center, std::array <int, 3>& xdims, std::array <int, 3>& dims, at::Tensor& src, at::Tensor& target);
	torch::Tensor makeCoord(std::array <int, 3>& dims);
	
	void retrievalByOptimalAssignment(at::Tensor& z_queries, std::vector<unsigned int>& survived, std::vector<std::string>& cadkey);
	void calculateRotationViaProcrustes(at::Tensor& noc, at::Tensor& mask, at::Tensor& scale, std::vector<std::array<float, 3>>& factor_interpolate, at::Tensor& grid2world, std::vector<Matrix3f>& rots, std::vector<std::string>cadkey_pred);
	
	
	std::vector<at::Tensor> feedForwardObjectDetection(float& res_scan, at::Tensor& sdf, at::Tensor& ft_pred);
	std::vector<CAD> feedForwardObject(Vox& v, torch::Tensor& ft_pred, torch::Tensor& ft_crop);
	
	
	Vector3f rotationMatrixToEulerAngles(Matrix4f& R);

	torch::jit::script::Module backbone, decode, feature2heatmap0, feature2descriptor, block0, feature2mask, feature2noc, feature2scale;
	std::unordered_map<std::string, at::Tensor> cadkey2latent, cadkey2sdf;
	std::vector<std::string> cadkey_pool;


	int batch_size = 1;
	double thresh_objectness = 0.5;
	int canonical_cube = 32;

	std::vector<std::array<float, 3>> factor_interpolate_collected;
	std::vector<Vector3i> obj_center_collected;
};