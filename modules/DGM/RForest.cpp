#include "RForest.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	Mat CRForest::predict(const Mat &featureVector)
	{
		// Assertion
		DGM_ASSERT(nclasses > 0);

		Mat res(m_nStates, 1, CV_32FC1, Scalar(0.0f));

		int max_nvotes = 0;
		cv::AutoBuffer<int> _votes(nclasses);
		int* votes = _votes;
		memset(votes, 0, sizeof(*votes) * nclasses);
		for (register int t = 0; t < ntrees; t++) {									// trees
			CvDTreeNode *predicted_node = trees[t]->predict(featureVector);
			int class_idx = predicted_node->class_idx;
			CV_Assert((class_idx >= 0) && (class_idx < nclasses));
			int nVotes = ++votes[class_idx];
			byte i = static_cast<byte>(predicted_node->value);
			res.at<float>(i, 0) = static_cast<float>(nVotes);
		} // t

		// Normalizing the potential
		float Sum = static_cast<float>(sum(res).val[0]);
		for (int s = 0; s < m_nStates; s++) 										// state
			res.at<float>(s, 0) /= Sum;

		return res;

	}

}
