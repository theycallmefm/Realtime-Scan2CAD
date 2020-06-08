# Real-time Scan2CAD Implementation using VoxelHashing

<img src="https://github.com/theycallmefm/VoxelHashing/blob/master/DepthSensingCUDA/proj-images/scene0470_chairs.gif" width="640" height="480" />
This project aligns a set of CAD models to the underlying scan in real-time using depth data. It uses two key approaches:

### End-to-End CAD Model Retrieval and 9DoF Alignment

This method takes a 3D scan and a set of CAD models as input and predicts a 9DoF pose that aligns each model to the underlying scan. End-to-End method is improvement upon Scan2CAD since it works in global fashion and allows for single forward pass.

**Paper:** [https://arxiv.org/pdf/1906.04201.pdf](https://arxiv.org/pdf/1906.04201.pdf)

**Code(Scan2CAD):** [https://github.com/skanti/Scan2CAD](https://github.com/skanti/Scan2CAD)

### VoxelHashing

This method is an online system for large and fine scale volumetric reconstruction based on a memory and speed efficient data structure. It has improved performance and reconstruction quality against other commonly known 3D reconstruction methods. 

**Paper:** [https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf](https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf)

**Code:**  [https://github.com/niessner/VoxelHashing](https://github.com/niessner/VoxelHashing)

## Method
The network has been trained using hashgrid data generated from VoxelHashing. Afterwards each module has been saved using torch's jit library(you can access my modules in \checkpoint folder but you can also train your own network and use it in this system) It uses libtorch to load modules. It takes the TSDF in camera view frustrum and aligns loaded CAD models according to the modules. You can look at the main structure below(Sorry for bad drawing...)

<img src="https://github.com/theycallmefm/VoxelHashing/blob/master/DepthSensingCUDA/proj-images/method.PNG" width="900" height="480" />


**Some notes:** 
- For the time being the project only supports CAD models in Scan2CAD dataset. I have used DXUT to load meshes using sdkmesh formats. You can create your own sdkmesh files using this library: [https://github.com/microsoft/DirectXTK/wiki/Rendering-a-model](https://github.com/microsoft/DirectXTK/wiki/Rendering-a-model "https://github.com/microsoft/DirectXTK/wiki/Rendering-a-model")
-  There is also something wrong with camera matrix. After first few frames you might see some objects floating around. I'll try to fix this in future.
-  Scan2CAD works every 5 frames after you activate it.
- I couldn't try this project in real-time at the time being. (Thanks corona) However it should be working on real-time.

### Requirements
The code was developed under VS2019.
- DirectX SDK June 2010
- Kinect SDK (prev. to 2.0)
- NVIDIA CUDA 10.1
- Libtorch 1.5.0

Feel free to ping me anytime. I might have forgotten to add some libraries/paths to project.

**Email:** fmert.algan@tum.de
