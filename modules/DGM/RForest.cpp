#include "RForest.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	Mat CRForest::predict(const Mat &featureVector)
	{
		// Assertion
		DGM_ASSERT(!roots.empty());

		if (m_nStates == 0) m_nStates = static_cast<byte>(classLabels.size());	// org:nclasses
		int	 ntrees  = static_cast<int>(roots.size());

		Mat res(m_nStates, 1, CV_32FC1, Scalar(0.0f));

		bool iscls = isClassifier();
		float scale = (!iscls) ? 1.0f / roots.size() : 1.0f;

		for (register int t = 0; t < ntrees; t++) {				// trees
			int s = static_cast<int>(predictTrees(Range(t, t + 1), featureVector, 0) * scale);
			res.at<float>(s, 0)++;
		} // t

		//res.setTo(0.01f);
		//int st = (int)CRTrees::predict(featureVector);
		//res.at<float>(st, 0) = 0.99f;

		if (ntrees) res /= ntrees;

		return res;
	}

}
