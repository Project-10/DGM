#include "TrainEdge.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
Mat CTrainEdge::getEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, float *params, size_t params_len, float weight) const
{
	// Assertions
	DGM_ASSERT_MSG(params, "Parameters are missing");

	Mat res = calculateEdgePotentials(featureVector1, featureVector2, params, params_len);
	if (weight != 1.0) pow(res, weight, res);

	// Normalization
	for (byte y = 0; y < m_nStates; y++) {
		float *pRes = res.ptr<float>(y);
		float  Sum = 0;
		for (byte x = 0; x < m_nStates; x++) Sum += pRes[x];
		if (Sum == 0) continue;
		for (byte x = 0; x < m_nStates; x++) pRes[x] *= 100 / Sum;
	} // y
	
	return res;
}

}
