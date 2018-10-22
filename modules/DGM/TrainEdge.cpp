#include "TrainEdge.h"
#include "GraphPairwiseExt.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	Mat CTrainEdge::getEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, const vec_float_t &vParams, float weight) const
	{
		Mat res = calculateEdgePotentials(featureVector1, featureVector2, vParams);
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
    
    // returns the matrix filled with ones, except the diagonal values wich are set to <values>
    Mat CTrainEdge::getDefaultEdgePotentials(const vec_float_t &values)
    {
        size_t nStates = values.size();
        Mat res(static_cast<int>(nStates), static_cast<int>(nStates), CV_32FC1, Scalar(1.0f));
        for (byte s = 0; s < nStates; s++) res.at<float>(s, s) = values[s];
        return res;
    }
}
