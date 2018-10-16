#include "TrainEdgePottsCS.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Calculates the Euclidean distance betwee two feature vectors
float calculateContrast(const Mat &featureVector1, const Mat &featureVector2)
{
	Mat		fv1, fv2, dfv;
	float	res = 0.0f;
	int nFeatures = featureVector1.rows;
	
	featureVector1.convertTo(fv1, CV_32FC1);
	featureVector2.convertTo(fv2, CV_32FC1);

	// distance between feature vectors
	subtract(fv1, fv2, dfv);				// dfv = fv1 - fv2;
	multiply(dfv, dfv, dfv);				// sqr(dfv);
	for (int i = 0; i < nFeatures; i++) 
		res += dfv.at<float>(i, 0);
	res = sqrtf(res / nFeatures);

	fv1.release();
	fv2.release();
	dfv.release();	

	return res;
}

// Charbonnier Penalizer
float penalizerChar(float x, float l)
{
	float res = l / sqrt(l*l + x*x);
	return MAX(FLT_EPSILON, res);
}

// Perona - Malik Penaliter
float penalizerPM(float x, float l)
{
	float res = l*l / (l*l + x*x);
	return MAX(FLT_EPSILON, res);
}

// Exponential penalizer
float penalizerExp (float x, float l)
{
	float res = expf(-l * x*x);			
	return MAX(FLT_EPSILON, res);
}


Mat	CTrainEdgePottsCS::calculateEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, const vec_float_t &vParams) const
{
	DGM_ASSERT_MSG((vParams.size() == 2) || (vParams.size() == m_nStates + 1), "Wrong number of parameters: %zu. It must be either %d or %u", vParams.size(), 2, m_nStates + 1);

	Mat res = CTrainEdgePotts::calculateEdgePotentials(featureVector1, featureVector2, vec_float_t(vParams.begin(), vParams.end() - 1));
	if (featureVector1.empty() || featureVector2.empty()) return res;	// no cotrast could be calcilated -> return potts edge potential

	// Assertions:
	DGM_ASSERT_MSG((featureVector1.type() == CV_8UC1) && (featureVector2.type() == CV_8UC1), 
		"One (or both) of input feature vectors has either wrong depth or more than one channel");
	DGM_ASSERT_MSG((featureVector1.size().width == 1) && (featureVector1.size().height == m_nFeatures), 
		"The first input feature vector has wrong size:(%d, %d)", featureVector1.size().width, featureVector1.size().height);
	DGM_ASSERT_MSG((featureVector2.size().width == 1) && (featureVector2.size().height == m_nFeatures), 
		"The second input feature vector has wrong size:(%d, %d)", featureVector2.size().width, featureVector2.size().height);

	float penalty;
	float dst = calculateContrast(featureVector1, featureVector2);
	switch(m_penApproach) {
		case eP_APP_PEN_CHAR:	penalty = penalizerChar(dst, vParams.back());	break;
		case eP_APP_PEN_PM:		penalty = penalizerPM(dst, vParams.back());		break;
		case eP_APP_PEN_EXP:	penalty = penalizerExp(dst, vParams.back());	break;
	}

	for (byte s = 0; s < m_nStates; s++) res.at<float>(s, s) = MAX(1.0f, res.at<float>(s, s) * penalty);

	return res;
}
}