#include "TrainNode.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constructor
CTrainNode::CTrainNode(byte nStates, word nFeatures)
    : CBaseRandomModel(nStates)
    , ITrain(nStates, nFeatures)
    , m_mask(Mat(nStates, 1, CV_8UC1))
{}

// Destructor
CTrainNode::~CTrainNode(void) 
{ }
	
void CTrainNode::addFeatureVecs(const Mat &featureVectors, const Mat &gt)
{
	DGM_ASSERT_MSG(featureVectors.channels() == m_nFeatures, "Number of features in the <featureVectors> (%d) does not correspond to the specified (%d)", featureVectors.channels(), m_nFeatures);
	DGM_VECTORWISE1<CTrainNode, &CTrainNode::addFeatureVec>(*this, featureVectors, gt);
}

void CTrainNode::addFeatureVecs(const vec_mat_t &featureVectors, const Mat &gt)
{
	DGM_ASSERT_MSG(featureVectors.size() == m_nFeatures, "Number of features in the <featureVectors> (%zu) does not correspond to the specified (%d)", featureVectors.size(), m_nFeatures);
	DGM_VECTORWISE1<CTrainNode, &CTrainNode::addFeatureVec>(*this, featureVectors, gt);
}

Mat	CTrainNode::getNodePotentials(const Mat &featureVectors, const Mat &weights, float Z) const
{
	// Assertions
	DGM_ASSERT_MSG(featureVectors.channels() == m_nFeatures, "Number of features in the <featureVectors> (%d) does not correspond to the specified (%d)", featureVectors.channels(), m_nFeatures);
	DGM_ASSERT(featureVectors.depth() == CV_8U);
	if (!weights.empty()) {
		DGM_ASSERT(featureVectors.size() == weights.size());
		DGM_ASSERT(weights.type() == CV_32FC1);
	}

	Mat res(featureVectors.size(), CV_32FC(m_nStates));
#ifdef ENABLE_PPL
	concurrency::parallel_for(0, res.rows, [&] (int y) {
		Mat pot;
		Mat vec(m_nFeatures, 1, CV_8UC1);
#else
	Mat pot;
	Mat vec(m_nFeatures, 1, CV_8UC1);
	for (int y = 0; y < res.rows; y++) {
#endif
		const byte  *pFv = featureVectors.ptr<byte>(y);
		const float *pW = weights.empty() ? NULL : weights.ptr<float>(y);
		float		*pRes = res.ptr<float>(y);
		for (int x = 0; x < res.cols; x++) {
			float weight = pW ? pW[x] : 1.0f;
			for (int f = 0; f < m_nFeatures; f++) vec.at<byte>(f, 0) = pFv[m_nFeatures * x + f];
			pot = getNodePotentials(vec, weight, Z);
			for (int s = 0; s < m_nStates; s++) pRes[m_nStates * x + s] = pot.at<float>(s, 0);
		} // x
	} // y	
#ifdef ENABLE_PPL
	);
#endif

	return res;
}

Mat	CTrainNode::getNodePotentials(const vec_mat_t &featureVectors, const Mat &weights, float Z) const
{
	DGM_ASSERT_MSG(featureVectors.size() == m_nFeatures, "Number of features in the <featureVectors> (%zu) does not correspond to the specified (%d)", featureVectors.size(), m_nFeatures);
	DGM_ASSERT(featureVectors[0].depth() == CV_8U);
	if (!weights.empty()) {
		DGM_ASSERT(featureVectors[0].size() == weights.size());
		DGM_ASSERT(weights.type() == CV_32FC1);
	}

	Mat res(featureVectors[0].size(), CV_32FC(m_nStates));
#ifdef ENABLE_PPL
	concurrency::parallel_for(0, res.rows, [&](int y) {
		Mat pot;
		Mat vec(m_nFeatures, 1, CV_8UC1);
#else
	Mat pot;
	Mat vec(m_nFeatures, 1, CV_8UC1);
	for (int y = 0; y < res.rows; y++) {
#endif
		const byte  **pFv = new const byte *[m_nFeatures];
		for (word f = 0; f < m_nFeatures; f++) pFv[f] = featureVectors[f].ptr<byte>(y);
		const float *pW = weights.empty() ? NULL : weights.ptr<float>(y);
		float		*pRes = res.ptr<float>(y);
		for (int x = 0; x < res.cols; x++) {
			float weight = pW ? pW[x] : 1.0f;
			for (int f = 0; f < m_nFeatures; f++) vec.at<byte>(f, 0) = pFv[f][x];
			pot = getNodePotentials(vec, weight, Z);
			for (int s = 0; s < m_nStates; s++) pRes[m_nStates * x + s] = pot.at<float>(s, 0);
		} // x
		delete[] pFv;
	} // y	
#ifdef ENABLE_PPL
	);
#endif

	return res;
}

Mat CTrainNode::getNodePotentials(const Mat &featureVector, float weight, float Z) const
{
	// Assertions
	DGM_ASSERT_MSG(featureVector.type() == CV_8UC1, 
		"The input feature vector has either wrong depth or more than one channel");
	DGM_ASSERT_MSG((featureVector.size().width == 1) && (featureVector.size().height == m_nFeatures), 
		"The input feature vector has wrong size:(%d, %d)", featureVector.size().width, featureVector.size().height);
	
	Mat res(m_nStates, 1, CV_32FC1, Scalar(0));
	const_cast<Mat &>(m_mask).setTo(1);
	calculateNodePotentials(featureVector, res, const_cast<Mat &>(m_mask));
	if (weight != 1.0f) pow(res, weight, res);

	// Normalization
	float Sum = static_cast<float>(sum(res).val[0]);
	if (Sum < FLT_EPSILON) {
		res.setTo(FLT_EPSILON, m_mask);		// Case of too small potentials (make all the cases equaly small probable)
	} else {
		if (Z > FLT_EPSILON)
			res *= 100.0 / Z;
		else 
			res *= 100.0 / Sum;
	}

	return res;
}
}
