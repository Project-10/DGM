#include "PriorNode.h"
#include "macroses.h"

namespace DirectGraphicalModels
{

void CPriorNode::addNodeGroundTruth(const Mat &gt)
{
	DGM_ELEMENTWISE1<CPriorNode, &CPriorNode::addNodeGroundTruth>(*this, gt);
}
	
void CPriorNode::addNodeGroundTruth(byte gt)
{
	DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %d is out of range %d", gt, m_nStates);
	m_histogramPrior.at<int>(gt, 0)++;
}

Mat CPriorNode::calculatePrior(void) const
{
	Mat res;
	double Sum = sum(m_histogramPrior)[0];
	m_histogramPrior.convertTo(res, CV_32FC1, 1.0 / Sum);
	return res;
}
}