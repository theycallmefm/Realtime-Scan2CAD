#include "stdafx.h"
#include "Scan2CAD.h"


void Scan2CAD::create()
{
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available on Libtorch!" << std::endl;
	}
	else {
		std::cout << "CUDA is not available on Libtorch." << std::endl;
		return;
	}
	loadTestPool();
	loadModules();
	loadLatentSpace();
	
	
}

void Scan2CAD::destroy()
{
	cadkey2sdf.clear();
	cadkey2latent.clear();
	cadkey_pool.clear();
	factor_interpolate_collected.clear();
	obj_center_collected.clear();
}

void Scan2CAD::loadLatentSpace() {
	const std::string filename = GlobalScan2CADState::get().s_CADlatentSpacePath;
	for (int i = 0; i < cadkey_pool.size(); i++) {
		cadkey2latent[cadkey_pool[i]] = torch::zeros(512, torch::kFloat32);
	}

	int size_latent = 512; //TODO change this to re.sub
	std::fstream f;
	f.clear();
	f.open(filename, std::fstream::in | std::fstream::binary);
	int32_t size;
	char key[41];
	float value[512]; 
	auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
	f.read((char*)&size, sizeof(int32_t));
	for (int i = 0; i < size; i++) {
		f.read((char*)&key, 41 * sizeof(char));
		f.read((char*)&value, size_latent * sizeof(float));
		std::string key_s = "";
		for (int j = 0; j < 41; j++) {
			key_s = key_s + key[j];
		}//TODO this is really bad way.. make something smarter
		//remove white spaces
		key_s.erase(remove(key_s.begin(), key_s.end(), ' '), key_s.end());
		if (std::find(cadkey_pool.begin(), cadkey_pool.end(), key_s) != cadkey_pool.end())
		{
			at::Tensor valueTensor = torch::zeros(512, torch::kFloat32);
			std::memcpy(valueTensor.data_ptr(), value, sizeof(float) * valueTensor.numel());
			cadkey2latent[key_s] = valueTensor;
		}
	}
	
}

at::Tensor Scan2CAD::loadCADsdf(std::vector<std::string>& cadkeys) {
	std::vector<torch::Tensor> tensor_list;
	for (int i = 0; i < cadkeys.size(); i++) {

		//std::cout << "cad " << cadkeys[i].substr(0, 8) <<" key "<< cadkeys[i].substr(9, cadkeys[i].length()) <<std::endl;
		std::string filename_cad_vox = GlobalScan2CADState::get().s_CADsdfPath+"/" + cadkeys[i].substr(0, 8) + "/" + cadkeys[i].substr(9, cadkeys[i].length()) + ".vox";
		//std::cout << filename_cad_vox << std::endl;
		Vox vox_cad = load_vox(filename_cad_vox, true);
		tensor_list.push_back(vox_cad.sdf);
		
		
	}
	return torch::stack({ tensor_list });
}

void Scan2CAD::loadCADsdfAll() {
	std::string csv_name = GlobalScan2CADState::get().s_cadsCSV;
	std::ifstream       file(csv_name.c_str());
	std::vector<std::string>   row;
	std::string                line;
	std::string                cell;
	std::unordered_map<std::string, at::Tensor> cadkey2sdf;
	bool isHeader = true;
	while (file)
	{
		std::getline(file, line);
		std::stringstream lineStream(line);
		row.clear();

		while (std::getline(lineStream, cell, ',')) {
			row.push_back(cell);

		}
		if (!isHeader && !row.empty()) {
			std::string filename_cad_vox = GlobalScan2CADState::get().s_CADsdfPath + "/" + row[0] + "/" + row[1] + ".vox";
			std::string cadkey = row[0] + "+" + row[1];
			Vox vox_cad = load_vox(filename_cad_vox, true);
			cadkey2sdf[cadkey] = vox_cad.sdf;
				//std::cout << "got the chair sdf" << std::endl;
			
		}
		isHeader = false;
		//break;
	}

}

void Scan2CAD::loadLatentSpaceAll() {
	const std::string filename = GlobalScan2CADState::get().s_CADlatentSpacePath;
	int size_latent = 512; //TODO change this to re.sub
	std::fstream f;
	f.clear();
	f.open(filename, std::fstream::in | std::fstream::binary);
	//std::string extension = filename.substr(filename.find_last_of(".") + 1);
	int32_t size;
	char key[41];
	float value[512]; //TODO change this to double maybe?
	auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
	f.read((char*)&size, sizeof(int32_t));
	//size = 1;
	for (int i = 0; i < size; i++) {
		f.read((char*)&key, 41 * sizeof(char));
		f.read((char*)&value, size_latent * sizeof(float));
		std::string key_s = "";
		for (int j = 0; j < 41; j++) {
			key_s = key_s + key[j];
		}//TODO this is really bad way.. make something smarter
		
		at::Tensor valueTensor = torch::zeros(512, torch::kFloat32);
		std::memcpy(valueTensor.data_ptr(), value, sizeof(float) * valueTensor.numel());
		key_s.erase(remove(key_s.begin(), key_s.end(), ' '), key_s.end()); //remove white spaces
		cadkey2latent[key_s] = valueTensor;
		//std::cout << "got the chair" << std::endl;
		
	}
	f.close();
}

void Scan2CAD::loadModules() {
	try {
		
		backbone = torch::jit::load(GlobalScan2CADState::get().s_backbone);
		decode = torch::jit::load(GlobalScan2CADState::get().s_decode);
		feature2heatmap0 = torch::jit::load(GlobalScan2CADState::get().s_feature2heatmap0);
		feature2descriptor = torch::jit::load(GlobalScan2CADState::get().s_feature2descriptor);
		block0 = torch::jit::load(GlobalScan2CADState::get().s_block0);
		feature2mask = torch::jit::load(GlobalScan2CADState::get().s_feature2mask);
		feature2noc = torch::jit::load(GlobalScan2CADState::get().s_feature2noc);
		feature2scale = torch::jit::load(GlobalScan2CADState::get().s_feature2scale);
		backbone.eval();
		decode.eval();
		feature2heatmap0.eval();
	}

	catch (const c10::Error& e) {
		std::cerr << "Error loading the models, please check checkpoint paths\n";
		return;
	}
}

void Scan2CAD::loadTestPool() {
	cadkey_pool = GlobalScan2CADState::get().s_cadkeypool;
	/*std::cout << "cadkey_pool" << std::endl;
	for (int i = 0; i < cadkey_pool.size(); i++) {
		std::cout << cadkey_pool[i] << std::endl;
	}*/
}

void Scan2CAD::nms(int kernel_size, at::Tensor& x) {
	//WARNING: this method could be deprecated in future
	auto x_max = torch::nn::Functional(torch::max_pool3d, kernel_size, 1, kernel_size / 2, 1, false)(x);
	auto x_binarized = torch::gt(x_max, thresh_objectness);
	x = torch::mul(x_binarized, torch::eq(x, x_max));
	//at::Tensor x_nms = torch::mul(x_binarized, torch::eq(x, x_max));
	//return x_nms;
}

void Scan2CAD::calcCenteredCropsAndCropCenterCopy(std::array <int, 3>& center, std::array <int, 3>& xdims, std::array <int, 3>& dims, at::Tensor& src, at::Tensor& target) {
	std::array <float, 3> h = { (dims[0] - 1) / 2, (dims[1] - 1) / 2, (dims[2] - 1) / 2 };
	
	std::array <int, 3> smin = { ceil(center[0] - h[0]), ceil(center[1] - h[1]), ceil(center[2] - h[2]) };
	std::array <int, 3> smax = { ceil(center[0] + h[0] + 1), ceil(center[1] + h[1] + 1), ceil(center[2] + h[2] + 1) };

	smin = { std::max(std::min(smin[0], xdims[0]), 0), std::max(std::min(smin[1], xdims[1]), 0), std::max(std::min(smin[2], xdims[2]), 0) };
	smax = { std::max(std::min(smax[0], xdims[0]), 0), std::max(std::min(smax[1], xdims[1]), 0), std::max(std::min(smax[2], xdims[2]), 0) };
	
	
	std::array <float, 3> deltamin = { center[0] - smin[0],center[1] - smin[1],center[2] - smin[2] };
	std::array <float, 3> deltamax = { smax[0] - center[0], smax[1] - center[1], smax[2] - center[2] };
	
	
	std::array <int, 3> tmin = { fmax(fmin(floor(h[0]) - deltamin[0], dims[0]), 0), fmax(fmin(floor(h[1]) - deltamin[1], dims[1]), 0), fmax(fmin(floor(h[2]) - deltamin[2], dims[2]), 0) };
	std::array <int, 3> tmax = { fmax(fmin(floor(h[0]) + deltamax[0], dims[0]), 0), fmax(fmin(floor(h[1]) + deltamax[1], dims[1]), 0), fmax(fmin(floor(h[2]) + deltamax[2], dims[2]), 0) };
	
	
	/*std::cout << "tmin: " << tmin[0] << " " << tmin[1] << " " << tmin[2] << std::endl;
	std::cout << "tmax: " << tmax[0] << " " << tmax[1] << " " << tmax[2] << std::endl;
	std::cout << "smin: " << smin[0] << " " << smin[1] << " " << smin[2] << std::endl;
	std::cout << "smax: " << smax[0] << " " << smax[1] << " " << smax[2] << std::endl;*/

	target[0].slice(1, tmin[0], tmax[0]).slice(2, tmin[1], tmax[1]).slice(3, tmin[2], tmax[2]) = src[0].slice(1, smin[0], smax[0]).slice(2, smin[1], smax[1]).slice(3, smin[2], smax[2]);
}

torch::Tensor Scan2CAD::makeCoord(std::array <int, 3>&  dims) {
	auto xcoords = torch::arange(dims[0], at::TensorOptions().dtype(torch::kFloat32)).to(at::Device(torch::kCUDA)).unsqueeze(1).unsqueeze(1).expand({ 1, dims[0], dims[1], dims[2] });
	auto ycoords = torch::arange(dims[1], at::TensorOptions().dtype(torch::kFloat32)).to(at::Device(torch::kCUDA)).unsqueeze(1).unsqueeze(0).expand({ 1, dims[0], dims[1], dims[2] });
	auto zcoords = torch::arange(dims[2], at::TensorOptions().dtype(torch::kFloat32)).to(at::Device(torch::kCUDA)).unsqueeze(0).unsqueeze(0).expand({ 1, dims[0], dims[1], dims[2] });

	return torch::cat({ zcoords,ycoords,xcoords }, 0);
}

void Scan2CAD::retrievalByOptimalAssignment(at::Tensor& z_queries,  std::vector<unsigned int>& survived, std::vector<std::string>& cadkey)
{
	at::Tensor hashmap_cads;
	std::vector<torch::Tensor> tensor_list;
	std::vector<std::vector<double>> cost_list;
	for (int i = 0; i < cadkey_pool.size(); i++) {
		tensor_list.push_back(cadkey2latent[cadkey_pool[i]]);
	}

	hashmap_cads = torch::stack({ tensor_list }).to(at::device(torch::kCUDA));
	for (int i = 0; i < z_queries.size(0); i++) {
		std::vector<double> vec;
		auto d = torch::norm(hashmap_cads - z_queries[i], 2, 1, false).to(torch::kDouble).to(torch::kCPU);
		vec.resize(cadkey_pool.size());
		std::memcpy(vec.data(), d.data_ptr<double>(), sizeof(double) * d.numel());
		cost_list.push_back(vec);
	}
	HungarianAlgorithm HungAlgo;
	std::vector<int> assignment;
	double cost = HungAlgo.Solve(cost_list, assignment);
	/*for (unsigned int x = 0; x < assignment.size(); x++) {
		std::cout << x << ": " << assignment[x] << ", ";

	}
	std::cout<<std::endl;*/

	for (unsigned int x = 0; x < cost_list.size(); x++) {
		if (assignment[x] != -1) {
			survived.push_back(x);
			cadkey.push_back(cadkey_pool[assignment[x]]);
		}

	}
}

void Scan2CAD::calculateRotationViaProcrustes(at::Tensor& noc, at::Tensor& mask, at::Tensor& scale, std::vector<std::array<float, 3>>& factor_interpolate, at::Tensor& grid2world, std::vector<Matrix3f>& rots, std::vector<std::string>cadkey_pred) {
	Timer t;
	auto n_batch_size = noc.size(0);
	/*if (n_batch_size == 0) {
		return torch::zeros(1);
	}*/
	assert(noc.size(1) == 3);
	std::vector<int64_t> dims = { noc.size(2),noc.size(3),noc.size(4) };//z y x
	noc = noc.view({ n_batch_size,3,-1 });
	mask = mask.view({ n_batch_size,1,-1 }).expand_as(noc);

	auto xcoords = torch::arange(dims[0], at::TensorOptions().dtype(torch::kFloat32)).to(at::Device(torch::kCUDA)).unsqueeze(1).unsqueeze(1).expand({ 1, dims[0], dims[1], dims[2] }).contiguous();
	auto ycoords = torch::arange(dims[1], at::TensorOptions().dtype(torch::kFloat32)).to(at::Device(torch::kCUDA)).unsqueeze(1).unsqueeze(0).expand({ 1, dims[0], dims[1], dims[2] }).contiguous();
	auto zcoords = torch::arange(dims[2], at::TensorOptions().dtype(torch::kFloat32)).to(at::Device(torch::kCUDA)).unsqueeze(0).unsqueeze(0).expand({ 1, dims[0], dims[1], dims[2] }).contiguous();
	auto coords = torch::cat({ xcoords,ycoords,zcoords }, 0).view({ 3,-1 });
	
	
	for (int i = 0; i < n_batch_size; i++)
	{
		auto a = noc[i].index({ mask[i] > 0.5 }).view({ 3,-1 }).to(at::Device(torch::kCUDA));
		a = torch::matmul(torch::diagflat(2.0 * scale[i]).to(at::Device(torch::kCUDA)), a);//scale to attain metric
		auto amean = torch::mean(a, 1).view({ 3,1 }).to(at::Device(torch::kCUDA));
		a = a - amean;
		
		auto b = coords.index({ (mask[i] > 0.5) }).view({ 3,-1 }).to(at::Device(torch::kCUDA)); //masked
		b = torch::matmul(torch::diagflat(4.0 / torch::tensor({ factor_interpolate[i][0],factor_interpolate[i][1],factor_interpolate[i][2] }, torch::TensorOptions().device(torch::kCUDA))), b).to(at::Device(torch::kCUDA)); //scale to attain a metric
		auto index_b = torch::zeros(3).to(torch::kLong).to(at::Device(torch::kCUDA));
		index_b[0] = long(2);
		index_b[1] = long(1);
		b = torch::index_select(b, 0, index_b).to(at::Device(torch::kCUDA)); //from zyx to xyz
		b = torch::matmul(grid2world.slice(0, 0, 3, 1).slice(1, 0, 3, 1).to(at::Device(torch::kCUDA)), b);
		auto bmean = torch::mean(b, 1).view({ 3,1 }).to(at::Device(torch::kCUDA));
		b = b - bmean;

		rots[i] = Matrix3f::Identity();//identity matrix
		
		//find R that maps from a to b
		if (a.size(1) < 6 || b.size(1) < 6) { //at least n residuals
			continue;
		}
		auto cov = torch::matmul(a, b.t());
		//switch to eigen for computational purposes
		Eigen::Matrix3f m;
		m << cov[0][0].item<float>(), cov[0][1].item<float>(), cov[0][2].item<float>(), cov[1][0].item<float>(), cov[1][1].item<float>(), cov[1][2].item<float>(), cov[2][0].item<float>(), cov[2][1].item<float>(), cov[2][2].item<float>();
		Eigen::JacobiSVD<Matrix3f> svd(m, ComputeThinU | ComputeThinV);
		
		auto UVt = svd.matrixU()* svd.matrixV().transpose();
		auto det = UVt.determinant();

		DiagonalMatrix<float, 3> E(Vector3f(1, 1, det));
		//auto R = svd.matrixV() * E *svd.matrixU().transpose();
		auto VE = svd.matrixV() * E;
		auto R = VE * svd.matrixU().transpose();
		/*std::cout << "cadkey_pred: " << std::endl << cadkey_pred[i] << std::endl;
		std::cout << "factor_interpolate" << std::endl << factor_interpolate[i][0]<<" " << factor_interpolate[i][1] << " " << factor_interpolate[i][2] << " " << std::endl;
		std::cout << "diag factor interpolate :" << std::endl << torch::diagflat(4.0 / torch::tensor({ factor_interpolate[i][0],factor_interpolate[i][1],factor_interpolate[i][2] }, torch::TensorOptions().device(torch::kCUDA))) << std::endl;
		std::cout << "grid2world" << std::endl << grid2world << std::endl;
		std::cout << "grid2world sliced: " << std::endl << grid2world.slice(0, 0, 3, 1).slice(1, 0, 3, 1).to(at::Device(torch::kCUDA)) << std::endl;
		std::cout << "amean" << std::endl << amean << std::endl;
		std::cout << "bmean" << std::endl << bmean << std::endl;
		std::cout << "cov" << std::endl << cov << std::endl;
		std::cout << "UVt" << std::endl << UVt << std::endl;
		std::cout << "det" << std::endl << det << std::endl;
		std::cout << "Ve" << std::endl << VE << std::endl;
		std::cout << "R" << std::endl << R << std::endl;*/
		rots[i] = R;
		//consistency[i] = (C * C).mean();

	}
	//std::cout << "Rotation calculation time: " << t.getElapsedTime() << std::endl;

}

std::vector<CAD> Scan2CAD::forward(HashData& hashData, const HashParams& hashParams) {
	int* dims;
	int* min_pos;
	dims = new int[3];
	min_pos = new int[3];
	float* sdf = createSDFTensor(hashData, hashParams, min_pos, dims);
	int n_elems = dims[0] * dims[1] * dims[2];
	float res = GlobalAppState::get().s_SDFVoxelSize;
	Timer t;
	Vox v = makeVoxFromSceneRepHashSDF(min_pos, dims, res, sdf);
	//std::cout << "sdf creation time: " << t.getElapsedTime() << std::endl;
	//vox.sdf = torch::from_blob(sdf.data(), { 1, 1, dims[2],dims[1],dims[0] }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
	
	Timer t2;
	torch::manual_seed(0);
	torch::NoGradGuard no_grad;
	std::vector<CAD> result;
	//Timer t,t2;
	auto ft_pred = backbone.forward({ v.sdf }).toTensor();
	//std::cout << "feature extraction time: " << t2.getElapsedTime() << std::endl;
	std::vector<at::Tensor> ft_crop_collected = feedForwardObjectDetection(v.res, v.sdf, ft_pred);
	if (ft_crop_collected.size()>0){
		torch::Tensor ft_crop = torch::cat({ ft_crop_collected });
		result = feedForwardObject(v,ft_pred,ft_crop);
	}
	//std::cout << "total elapsed time: " << t2.getElapsedTime() << std::endl;
	return result;

}

std::vector<at::Tensor> Scan2CAD::feedForwardObjectDetection(float&  res_scan, at::Tensor& sdf, at::Tensor& ft_pred) {
	Timer t;
	int confidence_radius = 9;//set to scannet 
	//torch::Tensor blur_kernel = gaussian3d(confidence_radius, 3).view({ 1,1,confidence_radius ,confidence_radius ,confidence_radius });

	
	std::array<int, 3>  dims_sdf = { sdf.size(2), sdf.size(3), sdf.size(4) };
	std::array<int, 3>  dims_ft = { ft_pred.size(2), ft_pred.size(3), ft_pred.size(4) };

	
	auto outputs = decode.forward({ ft_pred });
	auto x = outputs.toTuple()->elements()[0].toTensor();
	auto bbox_heatmap_pred = outputs.toTuple()->elements()[1].toTensor();

	
	auto object_heatmap = feature2heatmap0.forward({ x }).toTensor();
	
	
	object_heatmap = torch::nn::Sigmoid()(object_heatmap);

	nms(confidence_radius, object_heatmap);

	object_heatmap = object_heatmap.nonzero();
	//std::cout << "heatmap calculation elapsed time: " << t.getElapsedTime() << std::endl;

	object_heatmap = object_heatmap.slice(0, 0, 64);
	int64_t n_objects = object_heatmap.size(0);

	std::array<int, 3>  dims_crop = { canonical_cube,canonical_cube,canonical_cube };
	std::array<int, 3>  dims_ft_crop = { canonical_cube / 4 ,canonical_cube / 4 ,canonical_cube / 4 };
	//std::cout << "object detection elapsed time: " << t.getElapsedTime() << std::endl;
	Timer t2;
	//collector.resize(n_objects);
	//std::cout << "n_objects: " << n_objects << std::endl;
	std::vector<at::Tensor> ft_crop_collected;
	for (int j = 0; j < n_objects; j++) {
		//calculate croping center
		
		//TODO check objcenter shouldnt be float?
		std::array<int, 3>  c = { object_heatmap[j].slice(0, 2)[0].item<int>(), object_heatmap[j].slice(0, 2)[1].item<int>(), object_heatmap[j].slice(0, 2)[2].item<int>() };
		
		std::array<int, 3> c_ft;
		c_ft[0] = round(c[0]  * (dims_ft[0] - 1) / (dims_sdf[0] - 1));
		c_ft[1] = round(c[1] * (dims_ft[1] - 1) / (dims_sdf[1] - 1));
		c_ft[2] = round(c[2] * (dims_ft[2] - 1) / (dims_sdf[2] - 1));
		
		std::array<float, 3> bbox_pred = { bbox_heatmap_pred[0][0][c_ft[0]][c_ft[1]][c_ft[2]].item<float>(), bbox_heatmap_pred[0][1][c_ft[0]][c_ft[1]][c_ft[2]].item<float>(), bbox_heatmap_pred[0][2][c_ft[0]][c_ft[1]][c_ft[2]].item<float>() };
		
		
		std::array<int, 3> bbox_crop_ft = { ceil(ml::math::clamp((bbox_pred[0] / (res_scan * 4)), 4.0f, 24.0f)), ceil(ml::math::clamp((bbox_pred[1] / (res_scan * 4)), 4.0f, 24.0f)), ceil(ml::math::clamp((bbox_pred[2] / (res_scan * 4)), 4.0f, 24.0f)) };
		std::array<int, 3> bbox_crop = { bbox_crop_ft[0] * 4,bbox_crop_ft[1] * 4,bbox_crop_ft[2] * 4 };
		
		
		std::array<float, 3> factor_interpolate = { float(dims_crop[0]) / float(bbox_crop[0]), float(dims_crop[1]) / float(bbox_crop[1]) ,float(dims_crop[2]) / float(bbox_crop[2])};
		/////*std::cout << "c " << c[0] << " " << c[1] << " " << c[2] << std::endl;
		////std::cout << "dims_ft " << dims_ft[0] << " " << dims_ft[1] << " " << dims_ft[2] << std::endl;
		////std::cout << "dims_sdf " << dims_sdf[0] << " " << dims_sdf[1] << " " << dims_sdf[2] << std::endl;
		////std::cout << "c_ft " << c_ft[0] << " " << c_ft[1] << " " << c_ft[2] << std::endl;
		////std::cout << "bbox_pred " << bbox_pred[0] << " " << bbox_pred[1] << " " << bbox_pred[2] << std::endl;
		////std::cout << "dims_crop " << dims_crop[0] << " " << dims_crop[1] << " " << dims_crop[2] << std::endl;
		////std::cout << "bbox_crop " << bbox_crop[0] << " " << bbox_crop[1] << " " << bbox_crop[2] << std::endl;
		////std::cout << "factor_interpolate " << factor_interpolate[0] << " " << factor_interpolate[1] << " " << factor_interpolate[2] << std::endl;*/
		torch::Tensor ft_crop = torch::full({ 1,ft_pred.size(1), bbox_crop_ft[0], bbox_crop_ft[1], bbox_crop_ft[2] }, /*value=*/-1, at::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));

		/*std::cout << "calcCenteredCropsAndCropCenterCopy1" << std::endl << std::endl;*/
		//crop ft;
		calcCenteredCropsAndCropCenterCopy(c_ft, dims_ft, bbox_crop_ft, ft_pred, ft_crop);
		/*std::cout << "interpolate0" << std::endl << std::endl;*/
		ft_crop = torch::nn::functional::interpolate(ft_crop, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ dims_ft_crop[0], dims_ft_crop[1], dims_ft_crop[2] })).mode(torch::kTrilinear).align_corners(false));


		//sdf_crop
		torch::Tensor sdf_crop = torch::full({ 1, 1, bbox_crop[0], bbox_crop[1], bbox_crop[2] }, /*value=*/-0.15).to(at::Device(torch::kCUDA));
		/*std::cout << "calcCenteredCropsAndCropCenterCopy2" << std::endl << std::endl;*/
		calcCenteredCropsAndCropCenterCopy(c, dims_sdf, bbox_crop, sdf, sdf_crop);

		if (sdf_crop.max().item<float>() == float(-0.15)) {
			continue;
		}
		/*std::cout << "interpolate" << std::endl << std::endl;*/
		sdf_crop = torch::nn::functional::interpolate(sdf_crop, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ dims_ft_crop[0], dims_ft_crop[1], dims_ft_crop[1] })).mode(torch::kNearest));
		auto coords = makeCoord(bbox_crop).unsqueeze(0) * res_scan;
		/*std::cout << "interpolate2" << std::endl << std::endl;*/
		coords = torch::nn::functional::interpolate(coords, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ dims_ft_crop[0], dims_ft_crop[1], dims_ft_crop[1] })).mode(torch::kTrilinear).align_corners(false));
		auto coords2 = makeCoord(std::array <int, 3> {8,8,8}).unsqueeze(0) / 7.0;
		auto ft_crop_c = torch::cat({ ft_crop, sdf_crop, coords,coords2 }, 1);
		/*std::cout << "factor_interpolate_collected" << std::endl << std::endl;*/
		//TODO CHANGE OBJECT CENTER DEVICE TO CUDA
		/*std::unordered_map<std::string, at::Tensor> item = { {{"factor_interpolate", factor_interpolate},{"objcenter",torch::tensor({ c[0], c[1] ,c[1] })}, {"ft_crop" , ft_crop_c}} };
		collector.push_back(item);*/
		factor_interpolate_collected.push_back(factor_interpolate);
		Vector3i obj_center;
		obj_center << c[0], c[1], c[2];
		obj_center_collected.push_back(obj_center);
		ft_crop_collected.push_back(ft_crop_c);

	}
	//std::cout << "cropping elapsed time: " << t2.getElapsedTime() << std::endl;
	//TODO Add these checks
	//auto rng = std::default_random_engine{};
	//std::shuffle(std::begin(collector), std::end(collector), rng);
	/*if (collector.size() > 64) {
		collector = std::vector<std::unordered_map<std::string, at::Tensor>>(collector.begin(), collector.begin() + 64);
	}*/

	/*if (collector.size() == 1) {
		collector =
	}*/

	//int n_collector = ft_crop_collected.size();
	
	/*if (n_collector > 1) {
		std::vector<at::Tensor> ft_crop_cat;
		for (int i = 0; i < n_collector; i++) {
			ft_crop_cat.push_back(ft_crop_collected[i]);
		}
		ft_crop = torch::cat({ ft_crop_cat });
	}
	
	return ft_crop;*/
	//std::cout << "cropping features elapsed time: " << t2.getElapsedTime() << std::endl;
	return ft_crop_collected;
}

std::vector<CAD> Scan2CAD::feedForwardObject(Vox& v, torch::Tensor& ft_pred, torch::Tensor& ft_crop)
{
	Timer t,t2;
	std::vector<unsigned int> survived;
	std::vector<std::string> cadkey_pred;
	//std::vector<std::unordered_map<std::string, at::Tensor>> collector;
	std::vector<std::array<float, 3>> factor_interpolate_survived;
	std::vector<Vector3i> obj_center_survived;
	std::vector<CAD> output;
	auto grid2world =  torch::from_blob(v.grid2world.data(), { 4,4 }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
	//std::cout <<"obj_center_collected: " << obj_center_collected.size() << std::endl;
	if (obj_center_collected.size() > 0) {
		auto z_pred = feature2descriptor.forward({ ft_crop }).toTensor();
		retrievalByOptimalAssignment(z_pred, survived, cadkey_pred);
		//std::cout << "CAD retrieval elapsed time: " << t.getElapsedTime() << std::endl;
		std::vector<torch::Tensor> tensor_list;
		for (int i = 0; i < survived.size(); i++) {
			tensor_list.push_back(ft_crop[survived[i]]);
		}
		auto ft_crop = torch::stack({ tensor_list });


		//tensor_list.clear();
		for (int i = 0; i < survived.size(); i++) {
			//tensor_list.push_back(z_pred[survived[i]]);
			factor_interpolate_survived.push_back(factor_interpolate_collected[survived[i]]);
			obj_center_survived.push_back(obj_center_collected[survived[i]]);
		}
		//z_pred = torch::stack({ tensor_list });

		//tensor_list.clear();
		/*for (int i = 0; i < cadkey_pred.size(); i++) {
			tensor_list.push_back(cadkey2sdf[cadkey_pred[i]]);
		}
		at::Tensor cads = torch::stack({ tensor_list });*/
		at::Tensor cads = loadCADsdf(cadkey_pred);

		//tensor_list.clear();

		//writing forward_heads forward manually
		auto n_batch_size = ft_crop.size(0);
		double factor = canonical_cube / 128.0;
		std::vector<double> scale_factor = { factor ,factor ,factor };
		cads = torch::nn::functional::interpolate(cads, torch::nn::functional::InterpolateFuncOptions().scale_factor(scale_factor).mode(torch::kTrilinear).align_corners(false)).to(at::Device(torch::kCUDA));
		auto x = torch::cat({ ft_crop, cads }, 1);

		x = block0.forward({ x }).toTensor();
		auto mask_pred = feature2mask.forward({ x }).toTensor();
		auto noc_pred = feature2noc.forward({ x }).toTensor();
		x = torch::cat({ x,cads,noc_pred.detach(),mask_pred.detach() }, 1).to(at::Device(torch::kCUDA));

		auto scale_pred = feature2scale.forward({ x }).toTensor().view({ n_batch_size, 3 });
		//auto cadrec = decode_cad.forward({ z_pred.view({ n_batch_size,-1,4,4,4 }) }).toTensor();

		mask_pred = torch::nn::Sigmoid()(mask_pred);

		/*for (int i = 0; i < collector.size(); i++) {
			tensor_list.push_back(collector[i]["factor_interpolate"]);
		}*/
		//std::cout << "SOC prediction elapsed time:  " << t.getElapsedTime() << std::endl;
		//at::Tensor factor_interpolate = torch::stack({ tensor_list });
		std::vector<Matrix3f> rot_pred;
		rot_pred.resize(noc_pred.size(0));
		//auto consistency_pred = torch::empty({ noc_pred.size(0),1 }, torch::TensorOptions().dtype(torch::kFloat)).fill_(0).to(at::Device(torch::kCUDA));

		calculateRotationViaProcrustes(noc_pred, mask_pred, scale_pred, factor_interpolate_survived, grid2world, rot_pred, cadkey_pred);

		//std::cout << "obj_center_survived: " << obj_center_survived.size() << std::endl;
		//std::cout << "grid2world " << std::endl << v.grid2world << std::endl;
		for (int i = 0; i < obj_center_survived.size(); i++)
		{

			Vector3f T, S;
			RowVector4i xyz;
			Matrix4f transform, T_matrix, S_matrix, R, R_matrix;

			T_matrix = Matrix4f::Identity();
			S_matrix = Matrix4f::Identity();

			xyz << obj_center_survived[i](2), obj_center_survived[i](1), obj_center_survived[i](0), 1;//zyx --> xyz

			T(0) = v.grid2world(0,0) * xyz(0)+ v.grid2world(0, 1) * xyz(1)+ v.grid2world(0, 2) * xyz(2)+ v.grid2world(0, 3) * xyz(3);
			T(1) = v.grid2world(1, 0) * xyz(0) + v.grid2world(1, 1) * xyz(1) + v.grid2world(1, 2) * xyz(2) + v.grid2world(1, 3) * xyz(3);
			T(2) = v.grid2world(2, 0) * xyz(0) + v.grid2world(2, 1) * xyz(1) + v.grid2world(2, 2) * xyz(2) + v.grid2world(2, 3) * xyz(3);
			
			S(0) = scale_pred[i][0].item<float>();
			S(1) = scale_pred[i][1].item<float>();
			S(2) = scale_pred[i][2].item<float>();
			
			
			Vector4f position;
			position(0) = T(0), position(1) = T(1), position(2) = T(2), position(3) =1;
			T(1) = -position(2);
			T(2) = position(1);
			
			T_matrix(0, 3) = T(0), T_matrix(1, 3) = T(1), T_matrix(2, 3) = T(2);
			S_matrix(0, 0) = S(0), S_matrix(1, 1) = S(1), S_matrix(2, 2) = S(2);

			R(0,0)  = rot_pred[i](0, 0), R(0, 1) = rot_pred[i](0, 1), R(0, 2) = rot_pred[i](0, 2), R(0, 3) = 0,
			R(1, 0) = rot_pred[i](1, 0), R(1, 1) = rot_pred[i](1, 1), R(1, 2) = rot_pred[i](1, 2), R(1, 3) = 0,
			R(2, 0) = rot_pred[i](2, 0), R(2, 1) = rot_pred[i](2, 1), R(2, 2) = rot_pred[i](2, 2), R(2, 3) = 0,
			R(3, 0) = 0,				 R(3, 1) = 0,				  R(3, 2) = 0,				   R(3, 3) = 1;
		
			
			transform = T_matrix * R_matrix * S_matrix;
			Matrix4f transform2 = transform;
			transform(1, 0) = transform2(2, 0), transform(1, 1) = transform2(2, 1), transform(1, 2) = transform2(2, 2), transform(1, 3) = transform2(2, 3);
			transform(2, 0) = transform2(1, 0), transform(2, 1) = transform2(1, 1), transform(2, 2) = transform2(1, 2), transform(2, 3) = transform2(1, 3);

			output.push_back(CAD(cadkey_pred[i], T, R, S,transform));
		}

	}
	
	return output;

}

Vector3f Scan2CAD::rotationMatrixToEulerAngles(Matrix4f& R)
{

	assert(isRotationMatrix(R));

	float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(R(2, 1), R(2, 2));
		y = atan2(-R(2, 0), sy);
		z = atan2(R(1, 0), R(0, 0));
	}
	else
	{
		x = atan2(-R(1, 2), R(1, 1));
		y = atan2(-R(2, 0), sy);
		z = 0;
	}
	return Vector3f(x, y, z);



}