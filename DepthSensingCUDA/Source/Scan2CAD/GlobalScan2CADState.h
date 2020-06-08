#pragma once

/***********************************************************************************/
/* Global App state for Scan2CAD parameters  */
/***********************************************************************************/


#include "stdafx.h"

#include <vector>

#define X_GLOBAL_SCAN2CAD_STATE_FIELDS \
	X(std::string, s_CADlatentSpacePath) \
	X(std::string, s_CADsdfPath) \
	X(std::string, s_cadsCSV ) \
	X(std::string, s_backbone) \
	X(std::string, s_decode) \
	X(std::string, s_feature2heatmap0) \
	X(std::string, s_feature2descriptor) \
	X(std::string, s_block0) \
	X(std::string, s_feature2mask) \
	X(std::string, s_feature2noc) \
	X(std::string, s_feature2scale) \
	X(std::vector<std::string>, s_cadkeypool) \
	X(std::string, s_sdkmeshPath)

#ifndef VAR_NAME
#define VAR_NAME(x) #x
#endif


class GlobalScan2CADState
{
public:
#define X(type, name) type name;
	X_GLOBAL_SCAN2CAD_STATE_FIELDS
#undef X

		GlobalScan2CADState() {
	}

	//! setting default parameters
	

	//! sets the parameter file and reads
	void readMembers(const ParameterFile& parameterFile) {
		s_ParameterFile = parameterFile;
		readMembers();
	}

	//! reads all the members from the given parameter file (could be called for reloading)
	void readMembers() {
#define X(type, name) \
	if (!s_ParameterFile.readParameter(std::string(#name), name)) {MLIB_WARNING(std::string(#name).append(" ").append("uninitialized"));	name = type();}
		X_GLOBAL_SCAN2CAD_STATE_FIELDS
#undef X
	}

	//! prints all members
	/*void print() {
#define X(type, name) \
	std::cout << #name " = " << name << std::endl;
		X_GLOBAL_SCAN2CAD_STATE_FIELDS
#undef X
	}*/

	static GlobalScan2CADState& getInstance() {
		static GlobalScan2CADState s;
		return s;
	}

	static GlobalScan2CADState& get() {
		return getInstance();
	}
private:
	ParameterFile s_ParameterFile;
};
