#include "Infer.h"
#include "Decode.h"
#include "Graph.h"

namespace DirectGraphicalModels
{
	vec_byte_t CInfer::decode(unsigned int nIt, Mat &lossMatrix) 
	{
		if (nIt) infer(nIt);
		return CDecode::decode(getGraph(), lossMatrix);
	}
	
	vec_float_t CInfer::getConfidence(void) const
	{
		size_t nNodes = getGraph().getNumNodes();
		vec_float_t res(nNodes);
		Mat pot, srt;

		for (size_t n = 0; n < nNodes; n++) {						// all nodes
			getGraph().getNode(n, pot);
		
			sort(pot, srt, CV_SORT_EVERY_COLUMN | CV_SORT_DESCENDING);

			float max		 = srt.at<float>(0, 0);
			float second_max = srt.at<float>(1, 0);

			res[n] = (max == 0) ? 0.0f :  1.0f - second_max / max;
		} // n
	
		return res;
	}

	vec_float_t CInfer::getPotentials(byte state) const 
	{
		size_t nNodes = getGraph().getNumNodes();
		vec_float_t res(nNodes);
		Mat pot;

		for (size_t n = 0; n < nNodes; n++) {						// all nodes
			getGraph().getNode(n, pot);
			res[n] = pot.at<float>(state, 0);
		} // n

		return res;
	}
}