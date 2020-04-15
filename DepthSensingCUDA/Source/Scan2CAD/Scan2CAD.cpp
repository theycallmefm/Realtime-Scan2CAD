
#include "stdafx.h"
#include "Scan2CAD.h"



void Scan2CAD::create()
{
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Training on GPU." << std::endl;
	}
	else {
		std::cout << "CUDA is not available." << std::endl;
		return;
	}
	const std::string folder = "C:/scannet";
	loadLatentSpace(folder);
	loadCADsdf(folder);
	loadModules(folder);
	loadTestPool(folder);
}

void Scan2CAD::destroy()
{
	cadkey2sdf.clear();
	cadkey2latent.clear();
	cadkey_pool.clear();
	collector.clear();
}

void Scan2CAD::loadLatentSpace(const std::string folder) {
	const std::string filename = folder + "/scannet_annotations/scannet_cads.latent512";
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
	}
	f.close();
}

void Scan2CAD::loadCADsdf(const std::string folder) {
	std::string csv_name = folder + "/scannet_annotations/scannet_cads.csv";
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
			std::string filename_cad_vox = folder + "/shapenet_vox/dim32/" + row[0] + "/" + row[1] + ".vox";
			Vox vox_cad = load_vox(filename_cad_vox, true);
			std::string cadkey = row[0] + "+" + row[1];
			cadkey2sdf[cadkey] = vox_cad.sdf;
		}
		isHeader = false;
	}

}

void Scan2CAD::loadModules(const std::string folder) {
	
	try {
		backbone = torch::jit::load(folder + "/checkpoint/backbone.pt");

		//model_object_detection = torch::jit::load(model_object_detection_name);
		decode = torch::jit::load(folder + "/checkpoint/decode.pt");
		feature2heatmap0 = torch::jit::load(folder + "/checkpoint/feature2heatmap0.pt");

		feature2descriptor = torch::jit::load(folder + "/checkpoint/feature2descriptor.pt");
		block0 = torch::jit::load(folder + "/checkpoint/block0.pt");
		feature2mask = torch::jit::load(folder + "/checkpoint/feature2mask.pt");
		feature2noc = torch::jit::load(folder + "/checkpoint/feature2noc.pt");
		feature2scale = torch::jit::load(folder + "/checkpoint/feature2scale.pt");
		//decode_cad = torch::jit::load(forward_heads_decode_cad_name);
	}

	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return;
	}
}

void Scan2CAD::loadTestPool(const std::string folder) {
	cadkey_pool.push_back("02747177+85d8a1ad55fa646878725384d6baf445");
	cadkey_pool.push_back("03001627+b4371c352f96c4d5a6fee8e2140acec9");
	cadkey_pool.push_back("03001627+2c03bcb2a133ce28bb6caad47eee6580");
	cadkey_pool.push_back("03001627+2c03bcb2a133ce28bb6caad47eee6580");
	cadkey_pool.push_back("03001627+235c8ef29ef5fc5bafd49046c1129780");
	cadkey_pool.push_back("04379243+142060f848466cad97ef9a13efb5e3f7");
	cadkey_pool.push_back("03001627+bdc892547cceb2ef34dedfee80b7006");
}


torch::Tensor Scan2CAD::nms(int kernel_size, at::Tensor x) {
	//WARNING: this method could be deprecated in future
	auto x_max = torch::nn::Functional(torch::max_pool3d, kernel_size, 1, kernel_size / 2, 1, false)(x);
	auto x_binarized = torch::gt(x_max, thresh_objectness);
	at::Tensor x_nms = torch::mul(x_binarized, torch::eq(x, x_max));
	return x_nms;
}

void Scan2CAD::calcCenteredCrops(at::Tensor& center, at::Tensor& xdims, at::Tensor& dims, at::Tensor& smin, at::Tensor& smax, at::Tensor& tmin, at::Tensor& tmax) {
	//TODO add assertion for xdims and center
	auto h = (dims - 1) / float(2.0);
	smin = torch::ceil(center - h).to(torch::kInt);
	smax = torch::ceil(center + h + 1).to(torch::kInt);

	smin = torch::max(torch::min(smin, xdims), torch::zeros(3).to(torch::kInt));
	smax = torch::max(torch::min(smax, xdims), torch::zeros(3).to(torch::kInt));

	auto deltamin = center - smin;
	auto deltamax = smax - center;

	tmin = torch::max(torch::min(torch::floor(h).to(torch::kInt) - deltamin.to(torch::kInt), dims), torch::zeros(3).to(torch::kInt));
	tmax = torch::max(torch::min(torch::floor(h).to(torch::kInt) + deltamax.to(torch::kInt), dims), torch::zeros(3).to(torch::kInt));


}

void Scan2CAD::cropCenterCopy(at::Tensor& smin, at::Tensor& smax, at::Tensor& src, at::Tensor& tmin, at::Tensor& tmax, at::Tensor& target) {

	target[0].slice(1, tmin[0].item<int>(), tmax[0].item<int>()).slice(2, tmin[1].item<int>(), tmax[1].item<int>()).slice(3, tmin[2].item<int>(), tmax[2].item<int>()) = src[0].slice(1, smin[0].item<int>(), smax[0].item<int>()).slice(2, smin[1].item<int>(), smax[1].item<int>()).slice(3, smin[2].item<int>(), smax[2].item<int>());

}

torch::Tensor Scan2CAD::makeCoord(at::Tensor dims) {
	auto xcoords = torch::arange(dims[0].item<int>(), at::TensorOptions().dtype(torch::kFloat32)).to(at::Device(torch::kCUDA)).unsqueeze(1).unsqueeze(1).expand({ 1, dims[0].item<int>(), dims[1].item<int>(), dims[2].item<int>() });
	auto ycoords = torch::arange(dims[1].item<int>(), at::TensorOptions().dtype(torch::kFloat32)).to(at::Device(torch::kCUDA)).unsqueeze(1).unsqueeze(0).expand({ 1, dims[0].item<int>(), dims[1].item<int>(), dims[2].item<int>() });
	auto zcoords = torch::arange(dims[2].item<int>(), at::TensorOptions().dtype(torch::kFloat32)).to(at::Device(torch::kCUDA)).unsqueeze(0).unsqueeze(0).expand({ 1, dims[0].item<int>(), dims[1].item<int>(), dims[2].item<int>() });

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
		auto d = torch::norm(hashmap_cads - z_queries[i], 2, 1, false).to(torch::kDouble).to(torch::kCPU);;
		vec.resize(cadkey_pool.size());
		std::memcpy(vec.data(), d.data_ptr<double>(), sizeof(double) * d.numel());
		cost_list.push_back(vec);
	}
	HungarianAlgorithm HungAlgo;
	std::vector<int> assignment;
	double cost = HungAlgo.Solve(cost_list, assignment);
	for (unsigned int x = 0; x < cost_list.size(); x++) {
		if (assignment[x] != -1) {
			survived.push_back(x);
			cadkey.push_back(cadkey_pool[assignment[x]]);
		}

	}


}

void Scan2CAD::calculateRotationViaProcrustes(at::Tensor& noc, at::Tensor& mask, at::Tensor& scale, at::Tensor& factor_interpolate, at::Tensor& grid2world, at::Tensor& rots) {
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
		b = torch::matmul(torch::diagflat(4.0 / factor_interpolate[i]), b).to(at::Device(torch::kCUDA)); //scale to attain a metric
		auto index_b = torch::zeros(3).to(torch::kLong).to(at::Device(torch::kCUDA));
		index_b[0] = long(2);
		index_b[1] = long(1);
		b = torch::index_select(b, 0, index_b).to(at::Device(torch::kCUDA)); //from zyx to xyz
		b = torch::matmul(grid2world.slice(0, 0, 3, 1).slice(1, 0, 3, 1).to(at::Device(torch::kCUDA)), b);
		auto bmean = torch::mean(b, 1).view({ 3,1 }).to(at::Device(torch::kCUDA));
		b = b - bmean;

		rots[i][0][0], rots[i][1][1], rots[i][2][2] = float(1); //identity matrix

		//find R that maps from a to b
		if (a.size(1) < 6 || b.size(1) < 6) { //at least n residuals
			continue;
		}
		auto cov = torch::matmul(a, b.t());
		auto usv = torch::svd(cov);
		auto u = std::get<0>(usv);
		auto s = std::get<1>(usv);
		auto v = std::get<2>(usv);
		auto condition_number = torch::abs(s[0] / s[2]);

		auto is_finite = torch::isfinite(condition_number).any();//TODO do finite check


		auto UVt = torch::matmul(v, u.t());
		auto det = torch::det(UVt).to(at::Device(torch::kCUDA));
		auto det_ones = torch::ones(3).to(torch::kFloat).to(at::Device(torch::kCUDA));
		det_ones[2] = det;
		auto E = torch::diagflat(det_ones).detach();
		auto R = torch::matmul(torch::matmul(v, E), u.t());
		is_finite = torch::isfinite(R).any();//TODO do finite check

		a = torch::matmul(R.detach(), a);
		auto C = a - b;
		rots[i] = R;
		//consistency[i] = (C * C).mean();

	}

}



std::unordered_map<std::string, at::Tensor> Scan2CAD::forward(Vox& v) {
	torch::manual_seed(0);
	torch::NoGradGuard no_grad;
	
	backbone.eval();
	v.sdf = torch::clamp(v.sdf, -0.15, 2.0);
	at::Tensor ft_pred = backbone.forward({ v.sdf }).toTensor();

	torch::Tensor ft_crop = feedForwardObjectDetection(v,ft_pred);
	return feedForwardObject(v, ft_pred, ft_crop);


}

torch::Tensor Scan2CAD::feedForwardObjectDetection(Vox& v, at::Tensor& ft_pred) {
	auto sdf = v.sdf;
	int confidence_radius = 9;//set to scannet 
	//torch::Tensor blur_kernel = gaussian3d(confidence_radius, 3).view({ 1,1,confidence_radius ,confidence_radius ,confidence_radius });
	float res_scan = v.res;

	torch::Tensor dims_sdf = torch::tensor({ sdf.size(2), sdf.size(3),sdf.size(4) }).to(torch::kInt);
	torch::Tensor dims_ft = torch::tensor({ ft_pred.size(2), ft_pred.size(3),ft_pred.size(4) }).to(torch::kInt);

	decode.eval();
	auto outputs = decode.forward({ ft_pred });
	auto x = outputs.toTuple()->elements()[0].toTensor();
	auto bbox_heatmap_pred = outputs.toTuple()->elements()[1].toTensor();

	feature2heatmap0.eval();
	auto heatmap0_pred = feature2heatmap0.forward({ x }).toTensor();

	//model_object_detection.eval();
	//auto outputs = model_object_detection.forward({ ft_pred });

	//auto heatmap0_pred = outputs.toTuple()->elements()[0].toTensor(); //checked
	//auto bbox_heatmap_pred = outputs.toTuple()->elements()[1].toTensor();
	//auto scan_rec_pred = outputs.toTuple()->elements()[2].toTensor();

	auto heatmap0 = torch::nn::Sigmoid()(heatmap0_pred);

	torch::Tensor heatmap_nms = nms(confidence_radius, heatmap0);

	auto objects = heatmap_nms.nonzero();

	objects = objects.slice(0, 0, 64);
	int64_t n_objects = objects.size(0);

	torch::Tensor dims_crop = torch::full(3, canonical_cube, at::TensorOptions().dtype(torch::kFloat));
	torch::Tensor dims_ft_crop = torch::full(3, canonical_cube / 4, at::TensorOptions().dtype(torch::kFloat));

	collector.resize(n_objects);
	for (int j = 0; j < n_objects; j++) {
		auto idx = objects[j];
		//calculate croping center
		auto c = torch::tensor({ idx.slice(0, 2)[0].item<float>(),idx.slice(0, 2)[1].item<float>(),idx.slice(0, 2)[2].item<float>() }).to(at::Device(torch::kCPU));
		auto c_ft = torch::round(torch::tensor({ c[0].item<float>() / (dims_sdf[0].item<float>() - 1) * (dims_ft[0].item<float>() - 1), c[1].item<float>() / (dims_sdf[1].item<float>() - 1) * (dims_ft[1].item<float>() - 1) ,c[2].item<float>() / (dims_sdf[2].item<float>() - 1) * (dims_ft[2].item<float>() - 1) }));

		auto bbox_pred = torch::tensor({ bbox_heatmap_pred[0][0][c_ft[0].item<int>()][c_ft[1].item<int>()][c_ft[2].item<int>()].item<float>(), bbox_heatmap_pred[0][1][c_ft[0].item<int>()][c_ft[1].item<int>()][c_ft[2].item<int>()].item<float>(), bbox_heatmap_pred[0][2][c_ft[0].item<int>()][c_ft[1].item<int>()][c_ft[2].item<int>()].item<float>() });

		auto bbox_crop_ft = torch::ceil(torch::clamp(torch::div(bbox_pred, res_scan * 4), 4, 24)).to(torch::kInt); //<-- clamp between [...] voxels, add padding
		auto bbox_crop = torch::mul(bbox_crop_ft, 4);// match little crop

		auto factor_interpolate = torch::div(dims_crop, bbox_crop).to(at::Device(torch::kCUDA));

		torch::Tensor ft_crop = torch::full({ 1,ft_pred.size(1), bbox_crop_ft[0].item<int>(), bbox_crop_ft[1].item<int>(), bbox_crop_ft[2].item<int>() }, /*value=*/-1, at::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));


		//crop ft;
		at::Tensor smin, smax, tmin, tmax = torch::ones(3, at::TensorOptions().dtype(torch::kInt));

		calcCenteredCrops(c_ft, dims_ft, bbox_crop_ft, smin, smax, tmin, tmax);
		cropCenterCopy(smin, smax, ft_pred, tmin, tmax, ft_crop);
		//OK

		ft_crop = torch::nn::functional::interpolate(ft_crop, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ dims_ft_crop[0].item<int64_t>(), dims_ft_crop[1].item<int64_t>(), dims_ft_crop[2].item<int64_t>() })).mode(torch::kTrilinear).align_corners(false));


		//sdf_crop
		calcCenteredCrops(c, dims_sdf, bbox_crop, smin, smax, tmin, tmax);
		torch::Tensor sdf_crop = torch::full({ 1, 1, bbox_crop[0].item<int>(), bbox_crop[1].item<int>(), bbox_crop[2].item<int>() }, /*value=*/-0.15).to(at::Device(torch::kCUDA));
		cropCenterCopy(smin, smax, sdf, tmin, tmax, sdf_crop);


		if (sdf_crop.max().item<float>() == float(-0.15)) {
			continue;
		}

		sdf_crop = torch::nn::functional::interpolate(sdf_crop, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ dims_ft_crop[0].item<int64_t>(), dims_ft_crop[1].item<int64_t>(), dims_ft_crop[2].item<int64_t>() })).mode(torch::kNearest));
		auto coords = makeCoord(bbox_crop).unsqueeze(0) * res_scan;
		coords = torch::nn::functional::interpolate(coords, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ dims_ft_crop[0].item<int64_t>(), dims_ft_crop[1].item<int64_t>(), dims_ft_crop[2].item<int64_t>() })).mode(torch::kTrilinear).align_corners(false));
		auto coords2 = makeCoord(torch::tensor({ 8, 8, 8 })).unsqueeze(0) / 7.0;
		auto ft_crop_c = torch::cat({ ft_crop, sdf_crop, coords,coords2 }, 1);
		//TODO CHANGE OBJECT CENTER DEVICE TO CUDA
		std::unordered_map<std::string, at::Tensor> item = { {{"factor_interpolate", factor_interpolate},{"objcenter",c}, {"ft_crop" , ft_crop_c}} };
		collector.push_back(item);

	}

	//auto rng = std::default_random_engine{};
	//std::shuffle(std::begin(collector), std::end(collector), rng);
	if (collector.size() > 64) {
		collector = std::vector<std::unordered_map<std::string, at::Tensor>>(collector.begin(), collector.begin() + 64);
	}

	/*if (collector.size() == 1) {
		collector =
	}*/

	int n_collector = collector.size();
	std::cout << "n_collector: " << collector.size() << std::endl;
	at::Tensor ft_crop;
	if (n_collector > 1) {
		std::vector<at::Tensor> ft_crop_cat;
		for (int i = 0; i < n_collector; i++) {
			ft_crop_cat.push_back(collector[i]["ft_crop"]);
		}
		ft_crop = torch::cat({ ft_crop_cat });
	}

	return ft_crop;
}

std::unordered_map<std::string, at::Tensor> Scan2CAD::feedForwardObject(Vox& v, torch::Tensor& ft_pred, torch::Tensor& ft_crop)
{
	//TODO make it more memory efficient
	std::vector<unsigned int> survived;
	std::vector<std::string> cadkey_pred;
	at::Tensor mask_pred, scale_pred, rot_pred, noc_pred;
	std::vector<std::unordered_map<std::string, at::Tensor>> collector;
	std::unordered_map<std::string, at::Tensor> output;
	if (collector.size() >= 1) {
		auto z_pred = feature2descriptor.forward({ ft_crop }).toTensor();
		retrievalByOptimalAssignment(z_pred, survived, cadkey_pred);
		std::vector<torch::Tensor> tensor_list;
		for (int i = 0; i < survived.size(); i++) {
			tensor_list.push_back(ft_crop[survived[i]]);
		}
		auto ft_crop = torch::stack({ tensor_list });


		//tensor_list.clear();
		for (int i = 0; i < survived.size(); i++) {
			//tensor_list.push_back(z_pred[survived[i]]);
			collector.push_back(collector[survived[i]]);
		}
		//z_pred = torch::stack({ tensor_list });

		tensor_list.clear();
		for (int i = 0; i < cadkey_pred.size(); i++) {
			tensor_list.push_back(cadkey2sdf[cadkey_pred[i]]);
		}
		at::Tensor cads = torch::stack({ tensor_list });

		tensor_list.clear();

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

		for (int i = 0; i < collector.size(); i++) {
			tensor_list.push_back(collector[i]["factor_interpolate"]);
		}

		at::Tensor factor_interpolate = torch::stack({ tensor_list });
		auto rot_pred = torch::zeros({ noc_pred.size(0),3,3 }, torch::TensorOptions().dtype(torch::kFloat)).to(at::Device(torch::kCUDA));
		//auto consistency_pred = torch::empty({ noc_pred.size(0),1 }, torch::TensorOptions().dtype(torch::kFloat)).fill_(0).to(at::Device(torch::kCUDA));

		calculateRotationViaProcrustes(noc_pred, mask_pred, scale_pred, factor_interpolate, v.grid2world, rot_pred);


		for (int i = 0; i < collector.size(); i++)
		{

			at::Tensor T = torch::eye(4).to(at::Device(torch::kCUDA));
			at::Tensor R = torch::eye(4).to(at::Device(torch::kCUDA));
			at::Tensor S = torch::eye(4).to(at::Device(torch::kCUDA));
			
			T[0][3] = collector[i]["objcenter"].to(at::Device(torch::kCUDA))[0];
			T[1][3] = collector[i]["objcenter"].to(at::Device(torch::kCUDA))[1];
			T[2][3] = collector[i]["objcenter"].to(at::Device(torch::kCUDA))[2];
			
			R.slice(0, 0, 3).slice(1, 0, 3) = rot_pred[i];
			S.slice(0, 0, 3).slice(1, 0, 3) = torch::diag(scale_pred[i]);

			at::Tensor M = torch::matmul(torch::matmul(T, R), S);
			output[cadkey_pred[i]] = M;
			
		}

	}

	return output;

}
