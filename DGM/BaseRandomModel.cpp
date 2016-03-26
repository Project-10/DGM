#include "BaseRandomModel.h"
#include "macroses.h"

namespace DirectGraphicalModels
{

void CBaseRandomModel::save(const std::string &path, const std::string &name, short idx) const
{
	std::string fileName = generateFileName(path, name, idx); 
	FILE *pFile = fopen(fileName.c_str(), "wb");
	if (!pFile) {
		DGM_WARNING("Can't create file %s. Data was NOT saved.", fileName.c_str());
		return;
	}
	saveFile(pFile);
	fclose(pFile);
}

void CBaseRandomModel::load(const std::string &path, const std::string &name, short idx)
{
	std::string fileName = generateFileName(path, name, idx); 
	FILE *pFile = fopen(fileName.c_str(), "rb");
	DGM_ASSERT_MSG(pFile, "Can't load data from %s", fileName.c_str());
	loadFile(pFile);
	fclose(pFile);
}

std::string CBaseRandomModel::generateFileName(const std::string &path, const std::string &_name, short idx) const
{
	std::string name;
	if (_name.empty()) {
		std::string className = typeid(*this).name();
		name = className.substr(className.find("::") + 3);
	} else 
		name = _name;
	char str[7] = {0};
	if (idx >= 0) sprintf(str, "_%05d", idx);
	return path + name + str + ".dat";
}	
}