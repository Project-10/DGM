#include "TrainLink.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
	void CTrainLink::addFeatureVec(const Mat &featureVectors, const Mat &gtb, const Mat &gto)
	{
		DGM_ASSERT_MSG(featureVectors.channels() == m_nFeatures, "Number of features in the <featureVectors> (%d) does not correspond to the specified (%d)", featureVectors.channels(), m_nFeatures);
		DGM_VECTORWISE2<CTrainLink, &CTrainLink::addFeatureVec>(*this, featureVectors, gtb, gto);
	}
	
	void CTrainLink::addFeatureVec(const vec_mat_t &featureVectors, const Mat &gtb, const Mat &gto)
	{
		DGM_ASSERT_MSG(featureVectors.size() == m_nFeatures, "Number of features in the <featureVectors> (%zu) does not correspond to the specified (%d)", featureVectors.size(), m_nFeatures);
		DGM_VECTORWISE2<CTrainLink, &CTrainLink::addFeatureVec>(*this, featureVectors, gtb, gto);
	}

	Mat	CTrainLink::getLinkPotentials(const Mat &featureVector, float weight) const 
	{
		Mat res = calculateLinkPotentials(featureVector);
		if (weight != 1.0) pow(res, weight, res);
		return res;
	}

}