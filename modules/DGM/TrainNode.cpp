#include "TrainNode.h"

#include "TrainNodeNaiveBayes.h"
#include "TrainNodeGM.h"
#include "TrainNodeGMM.h"
#include "TrainNodeCvGM.h"
#include "TrainNodeCvGMM.h"
#include "TrainNodeKNN.h"
#include "TrainNodeCvKNN.h"
#include "TrainNodeCvRF.h"
#include "TrainNodeMsRF.h"
#include "TrainNodeCvANN.h"
#include "TrainNodeCvSVM.h"

#include "macroses.h"

namespace DirectGraphicalModels
{
	// Factory method
	std::shared_ptr<CTrainNode> CTrainNode::create(byte nodeRandomModel, byte nStates, word nFeatures)
	{
		switch (nodeRandomModel)
		{
		case NodeRandomModel::Bayes:	return std::make_shared<CTrainNodeBayes>(nStates, nFeatures);
		case NodeRandomModel::GM: 		return std::make_shared<CTrainNodeGM>(nStates, nFeatures);	
		case NodeRandomModel::GMM: 		return std::make_shared<CTrainNodeGMM>(nStates, nFeatures);	
		case NodeRandomModel::CvGM: 	return std::make_shared<CTrainNodeCvGM>(nStates, nFeatures);	
		case NodeRandomModel::CvGMM:	return std::make_shared<CTrainNodeCvGMM>(nStates, nFeatures);		
		case NodeRandomModel::KNN: 		return std::make_shared<CTrainNodeKNN>(nStates, nFeatures);	
		case NodeRandomModel::CvKNN:	return std::make_shared<CTrainNodeCvKNN>(nStates, nFeatures);		
		case NodeRandomModel::CvRF: 	return std::make_shared<CTrainNodeCvRF>(nStates, nFeatures);	
#ifdef USE_SHERWOOD
		case NodeRandomModel::MsRF: 	return std::make_shared<CTrainNodeMsRF>(nStates, nFeatures);	
#endif
		case NodeRandomModel::CvANN:	return std::make_shared<CTrainNodeCvANN>(nStates, nFeatures);		
		case NodeRandomModel::CvSVM:	return std::make_shared<CTrainNodeCvSVM>(nStates, nFeatures);		
		default:
			DGM_ASSERT_MSG(false, "Unknown type of the node random model");
		}
	}

	void CTrainNode::addFeatureVecs(const Mat &featureVectors, const Mat &gt)
	{
		DGM_ASSERT_MSG(featureVectors.channels() == getNumFeatures(), "Number of features in the <featureVectors> (%d) does not correspond to the specified (%d)", featureVectors.channels(), getNumFeatures());
		DGM_VECTORWISE1<CTrainNode, &CTrainNode::addFeatureVec>(*this, featureVectors, gt);
	}

	void CTrainNode::addFeatureVecs(const vec_mat_t &featureVectors, const Mat &gt)
	{
		DGM_ASSERT_MSG(featureVectors.size() == getNumFeatures(), "Number of features in the <featureVectors> (%zu) does not correspond to the specified (%d)", featureVectors.size(), getNumFeatures());
		DGM_VECTORWISE1<CTrainNode, &CTrainNode::addFeatureVec>(*this, featureVectors, gt);
	}

	Mat	CTrainNode::getNodePotentials(const Mat &featureVectors, const Mat &weights, float Z) const
	{
		// Assertions
		DGM_ASSERT_MSG(featureVectors.channels() == getNumFeatures(), "Number of features in the <featureVectors> (%d) does not correspond to the specified (%d)", featureVectors.channels(), getNumFeatures());
		DGM_ASSERT(featureVectors.depth() == CV_8U);
		if (!weights.empty()) {
			DGM_ASSERT(featureVectors.size() == weights.size());
			DGM_ASSERT(weights.type() == CV_32FC1);
		}

		Mat res(featureVectors.size(), CV_32FC(m_nStates));
#ifdef ENABLE_PPL
		concurrency::parallel_for(0, res.rows, [&] (int y) {
			Mat pot;
			Mat vec(getNumFeatures(), 1, CV_8UC1);
#else
		Mat pot;
		Mat vec(getNumFeatures(), 1, CV_8UC1);
		for (int y = 0; y < res.rows; y++) {
#endif
			const byte  *pFv = featureVectors.ptr<byte>(y);
			const float *pW = weights.empty() ? NULL : weights.ptr<float>(y);
			float		*pRes = res.ptr<float>(y);
			for (int x = 0; x < res.cols; x++) {
				float weight = pW ? pW[x] : 1.0f;
				for (int f = 0; f < getNumFeatures(); f++) vec.at<byte>(f, 0) = pFv[getNumFeatures() * x + f];
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
		DGM_ASSERT_MSG(featureVectors.size() == getNumFeatures(), "Number of features in the <featureVectors> (%zu) does not correspond to the specified (%d)", featureVectors.size(), getNumFeatures());
		DGM_ASSERT(featureVectors[0].depth() == CV_8U);
		if (!weights.empty()) {
			DGM_ASSERT(featureVectors[0].size() == weights.size());
			DGM_ASSERT(weights.type() == CV_32FC1);
		}

		Mat res(featureVectors[0].size(), CV_32FC(m_nStates));
#ifdef ENABLE_PPL
		concurrency::parallel_for(0, res.rows, [&](int y) {
			Mat pot;
			Mat vec(getNumFeatures(), 1, CV_8UC1);
#else
		Mat pot;
		Mat vec(getNumFeatures(), 1, CV_8UC1);
		for (int y = 0; y < res.rows; y++) {
#endif
			const byte  **pFv = new const byte *[getNumFeatures()];
			for (word f = 0; f < getNumFeatures(); f++) pFv[f] = featureVectors[f].ptr<byte>(y);
			const float *pW = weights.empty() ? NULL : weights.ptr<float>(y);
			float		*pRes = res.ptr<float>(y);
			for (int x = 0; x < res.cols; x++) {
				float weight = pW ? pW[x] : 1.0f;
				for (int f = 0; f < getNumFeatures(); f++) vec.at<byte>(f, 0) = pFv[f][x];
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
		DGM_ASSERT_MSG((featureVector.size().width == 1) && (featureVector.size().height == getNumFeatures()),
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
