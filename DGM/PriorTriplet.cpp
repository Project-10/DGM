#include "PriorTriplet.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
void CPriorTriplet::addTripletGroundTruth(byte gt1, byte gt2, byte gt3)
{
	DGM_ASSERT(gt1 < m_nStates);
	DGM_ASSERT(gt2 < m_nStates);
	DGM_ASSERT(gt3 < m_nStates);
	m_histogramPrior.at<int>(gt1, gt2, gt3)++;
}

///@todo Implement this function
///@warning This function is not implemented (returns only uniform distribution "all ones")
Mat CPriorTriplet::calculatePrior(void) const
{
	int size[] = {m_nStates, m_nStates, m_nStates};
	Mat res(3, size, CV_32FC1);

	res.setTo(1.0f);

	return res;
}

}