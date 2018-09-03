#include "Decode.h"
#include "Graph.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	vec_byte_t CDecode::decode(const CGraph &graph, Mat &lossMatrix)
	{
		size_t		nNodes		= graph.getNumNodes();			// number of nodes
		vec_byte_t	res(nNodes);
		Mat			pot;
		bool		ifLossMat	= !lossMatrix.empty();

		// Getting optimal state
		for (size_t n = 0; n < nNodes; n++) {						// all nodes
			graph.getNode(n, pot);
			if (ifLossMat) gemm(lossMatrix, pot, 1.0, Mat(), 0.0, pot);
		
			Point extremumLoc;
			if (ifLossMat) minMaxLoc(pot, NULL, NULL, &extremumLoc, NULL);
			else minMaxLoc(pot, NULL, NULL, NULL, &extremumLoc);
			res[n] = static_cast<byte>(extremumLoc.y);
		} // n

		return res;
	}

	Mat	CDecode::getDefaultLossMatrix(byte nStates)
	{
		Mat res(nStates, nStates, CV_32FC1, Scalar(1.0f));
		for (byte i = 0; i < nStates; i++) res.at<float>(i,i) = 0.0f;
		return res;
	}
}
