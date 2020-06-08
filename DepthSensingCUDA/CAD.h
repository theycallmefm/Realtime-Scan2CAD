#pragma once
#include <Eigen.h>

class CAD
{
	public:
		CAD(std::string cadkey, Vector3f T, Vector3f S, Vector3f R) {
			this->cadkey = cadkey;
			this->T = T;
			this->R = R;
			this->S = S;
		}
		~CAD() {}

		std::string getCADKey() {
			return cadkey;
		}

		Vector3f getTranslation() {
			return T;
		}

		Vector3f getRotation() {
			return R;
		}
	
		Vector3f getScale() {
			return S;
		}


	private:
		std::string cadkey; // typename+cad
		Vector3f T;
		Vector3f R; // euler angles X, Y, Z
		Vector3f S;

};