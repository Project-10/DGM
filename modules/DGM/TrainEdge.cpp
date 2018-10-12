#include "TrainEdge.h"
#include "GraphPairwiseExt.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	void CTrainEdge::addFeatureVecs(const Mat &featureVectors, const Mat &gt, const CGraphPairwiseExt &graph)
	{
		// Assertions
		DGM_ASSERT_MSG(featureVectors.size() == gt.size(), "The size of <featureVectors> does not correspond to the size of <gt>");
		DGM_ASSERT_MSG(featureVectors.depth() == CV_8U, "The argument <featureVectors> has wrong depth");
		DGM_ASSERT_MSG(gt.type() == CV_8UC1, "The argument <gt> has either wrong depth or more than one channel");
		DGM_ASSERT_MSG(featureVectors.channels() == m_nFeatures,
			"Number of features in the <featureVectors> (%d) does not correspond to the specified (%d)", featureVectors.channels(), m_nFeatures);

		const byte		graphType = graph.getType();
		const word		nFeatures = featureVectors.channels();

		Mat featureVector1(nFeatures, 1, CV_8UC1);
		Mat featureVector2(nFeatures, 1, CV_8UC1);

		for (int y = 0; y < gt.rows; y++) {
			const byte *pFV1 = featureVectors.ptr<byte>(y);
			const byte *pFV2 = y > 0 ? featureVectors.ptr<byte>(y - 1) : NULL;
			const byte *pGt1 = gt.ptr<byte>(y);
			const byte *pGt2 = y > 0 ? gt.ptr<byte>(y - 1) : NULL;
			for (int x = 0; x < gt.cols; x++) {
				for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFV1[nFeatures * x + f];					// featureVector[x][y]
				if (graphType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFV1[nFeatures * (x - 1) + f];		// featureVector[x-1][y]
						addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt1[x - 1]);
						addFeatureVecs(featureVector2, pGt1[x - 1], featureVector1, pGt1[x]);
					}
					if (y > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFV2[nFeatures * x + f];			// featureVector[x][y-1]
						addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt2[x]);
						addFeatureVecs(featureVector2, pGt2[x], featureVector1, pGt1[x]);
					}
				}
				if (graphType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFV2[nFeatures * (x - 1) + f];		// featureVector[x-1][y-1]
						addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt2[x - 1]);
						addFeatureVecs(featureVector2, pGt2[x - 1], featureVector1, pGt1[x]);
					}
					if ((x < gt.cols - 1) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFV2[nFeatures * (x + 1) + f];		// featureVector[x+1][y-1]
						addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt2[x + 1]);
						addFeatureVecs(featureVector2, pGt2[x + 1], featureVector1, pGt1[x]);
					}
				}
			} // x
		} // y
	}

	void CTrainEdge::addFeatureVecs(const vec_mat_t &featureVectors, const Mat &gt, const CGraphPairwiseExt &graph)
	{
		// Assertions
		DGM_ASSERT_MSG(featureVectors[0].size() == gt.size(), "The size of <featureVectors> does not correspond to the size of <gt>");
		DGM_ASSERT_MSG(featureVectors[0].type() == CV_8UC1, "The argument <featureVectors> has either wrong depth or more than one channel");
		DGM_ASSERT_MSG(gt.type() == CV_8UC1, "The argument <gt> has either wrong depth or more than one channel");
		DGM_ASSERT_MSG(featureVectors.size() == m_nFeatures,
			"Number of features in the <featureVectors> (%zu) does not correspond to the specified (%d)", featureVectors.size(), m_nFeatures);

		const byte		graphType = graph.getType();
		const word		nFeatures = static_cast<word>(featureVectors.size());

		Mat featureVector1(nFeatures, 1, CV_8UC1);
		Mat featureVector2(nFeatures, 1, CV_8UC1);
		
		std::vector<const byte *> vFV1(nFeatures);
		std::vector<const byte *> vFV2(nFeatures);
		for (int y = 0; y < gt.rows; y++) {
			for (word f = 0; f < nFeatures; f++) {
				vFV1[f] = featureVectors[f].ptr<byte>(y);
				if (y > 0) vFV2[f] = featureVectors[f].ptr<byte>(y - 1);
			}
			const byte *pGt1 = gt.ptr<byte>(y);
			const byte *pGt2 = y > 0 ? gt.ptr<byte>(y - 1) : NULL;
			for (int x = 0; x < gt.cols; x++) {
				for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = vFV1[f][x];					// featureVector[x][y]
				if (graphType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = vFV1[f][x - 1];		// featureVector[x-1][y]
						addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt1[x - 1]);
						addFeatureVecs(featureVector2, pGt1[x - 1], featureVector1, pGt1[x]);
					}
					if (y > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = vFV2[f][x];			// featureVector[x][y-1]
						addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt2[x]);
						addFeatureVecs(featureVector2, pGt2[x], featureVector1, pGt1[x]);
					}
				}
				if (graphType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = vFV2[f][x - 1];		// featureVector[x-1][y-1]
						addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt2[x - 1]);
						addFeatureVecs(featureVector2, pGt2[x - 1], featureVector1, pGt1[x]);
					}
					if ((x < gt.cols - 1) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = vFV2[f][x + 1];		// featureVector[x+1][y-1]
						addFeatureVecs(featureVector1, pGt1[x], featureVector2, pGt2[x + 1]);
						addFeatureVecs(featureVector2, pGt2[x + 1], featureVector1, pGt1[x]);
					}
				}
			} // x
		} // y
	}

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
    
    // returns the matrix filled with ones, except the diagonal values wich are set to <values>
    Mat CTrainEdge::getDefaultEdgePotentials(const vec_float_t &values)
    {
        size_t nStates = values.size();
        Mat res(static_cast<int>(nStates), static_cast<int>(nStates), CV_32FC1, Scalar(1.0f));
        for (byte s = 0; s < nStates; s++) res.at<float>(s, s) = values[s];
        return res;
    }
}
