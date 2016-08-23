#include "TrainEdgePotts.h"
#include "macroses.h"

namespace DirectGraphicalModels
{

// returns the matrix filled with ones, except the diagonal wich is set to <val>
Mat CTrainEdgePotts::getEdgePotentials(float val, byte nStates) 
{
	Mat res(nStates, nStates, CV_32FC1, Scalar(1.0f));
	for (byte s = 0; s < nStates; s++) res.at<float>(s, s) = val; 
	return res;
}

// returns the matrix filled with ones, except the diagonal values wich are set to <values>
Mat CTrainEdgePotts::getEdgePotentials(float *values, byte nStates) 
{
	Mat res(nStates, nStates, CV_32FC1, Scalar(1.0f));
	for (byte s = 0; s < nStates; s++) res.at<float>(s, s) = values[s]; 
	return res;
}

Mat CTrainEdgePotts::calculateEdgePotentials(const Mat &, const Mat &, float *params, size_t params_len) const
{
	DGM_ASSERT_MSG((params_len == 1) || (params_len == m_nStates), "Wrong number of parameters: %zu. It must be either %d or %u", params_len, 1, m_nStates);

	if (params_len == 1) return getEdgePotentials(params[0], m_nStates);
	else				 return getEdgePotentials(params, m_nStates);
}

}