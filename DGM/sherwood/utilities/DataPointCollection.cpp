#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
size_t DataPointCollection::Count(unsigned char state) const
{
	unsigned int res = 0;
	for (size_t i = 0; i < m_vLabels.size(); i++)
		if (m_vLabels.at(i) == state) res++;
	return res;
}

int DataPointCollection::GetIntegerLabel(int i) const 
{
	if (m_vLabels.size() == 0) throw std::runtime_error("Data have no associated class labels.");
	return m_vLabels[i]; // may throw an exception if index is out of range
}
} } }
