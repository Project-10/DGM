#include "TrainNode.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constructor
CTrainNode::CTrainNode(byte nStates, word nFeatures) 
	: ITrain(nStates, nFeatures)
	, CBaseRandomModel(nStates)
	, m_mask(Mat(nStates, 1, CV_8UC1))
{}

// Destructor
CTrainNode::~CTrainNode(void) 
{ }
	
void CTrainNode::addFeatureVec(const Mat &featureVectors, const Mat &gt)
{
	DGM_ASSERT_MSG(featureVectors.channels() == m_nFeatures, "Number of features in the <featureVectors> (%d) does not correspond to the specified (%d)", featureVectors.channels(), m_nFeatures);
	DGM_VECTORWISE1<CTrainNode, &CTrainNode::addFeatureVec>(*this, featureVectors, gt);
}

void CTrainNode::addFeatureVec(const vec_mat_t &featureVectors, const Mat &gt)
{
	DGM_ASSERT_MSG(featureVectors.size() == m_nFeatures, "Number of features in the <featureVectors> (%zu) does not correspond to the specified (%d)", featureVectors.size(), m_nFeatures);
	DGM_VECTORWISE1<CTrainNode, &CTrainNode::addFeatureVec>(*this, featureVectors, gt);
}

Mat CTrainNode::getNodePotentials(const Mat &featureVector, float weight) const
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
	if (Sum < FLT_EPSILON)
		res.setTo(FLT_EPSILON, m_mask);		// Case of too small potentials (make all the cases equaly small probable)
	//else
	//	res *= 100 / Sum;

	return res;
}
}