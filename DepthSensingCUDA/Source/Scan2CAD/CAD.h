#pragma once
#include <Eigen.h>
class CAD
{
	public:
		CAD(std::string cadkey, Vector3f T, Matrix4f R, Vector3f S, Matrix4f transform) {
			this->cadkey = cadkey;
			this->T = T;
			this->R = R;
			this->S = S;
			this->transform = transform;

			this->rotAngle(0) = atan2f(R(2,1), R(2, 2));
			this->rotAngle(1) = atan2f(-R(2, 0), sqrt(powf(R(2, 1),2)+ powf(R(2, 2), 2)));
			this->rotAngle(2) = atan2f(R(1, 0), R(0, 0));
		}
		~CAD() {}

		std::string getCADKey() {
			return cadkey;
		}

		Vector3f getTranslation() {
			return T;
		}

		Matrix4f getRotation() {
			return R;
		}
	
		Vector3f getScale() {
			return S;
		}

		Matrix4f getTransform() {
			return transform;
		}

		Vector3f getRotAngle() {
			return rotAngle;
		}

	private:
		std::string cadkey; // typename+cad
		Vector3f T;
		Matrix4f R; 
		Vector3f S;
		Matrix4f transform;
		Vector3f rotAngle;

};