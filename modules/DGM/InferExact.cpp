#include "InferExact.h"
#include "GraphPairwise.h"
#include <numeric>

namespace DirectGraphicalModels
{
	void CInferExact::infer(unsigned int)
	{
        size_t		nNodes  = CInfer::getGraph().getNumNodes();
        byte        nStates = CInfer::getGraph().getNumStates();
		vec_byte_t	state(nNodes);				
	
		// Calculating the potentials for every possible configuration
		vec_float_t	P = calculatePotentials();

		// Calculating the partition function
		float Z = std::accumulate(P.cbegin(), P.cend(), 0.0f);
	
		// Filling node potentials with marginal probabilities
        for (size_t n = 0; n < nNodes; n++)
            CInfer::getGraph().setNode(n, Mat::zeros(nStates, 1, CV_32FC1));
        setState(state, 0);
        Mat nPot;
        for(float &p : P) {
            for (size_t n = 0; n < nNodes; n++) {
                CInfer::getGraph().getNode(n, nPot);
                nPot.at<float>(state[n], 0) += p / Z;
                CInfer::getGraph().setNode(n, nPot);
            }
            incState(state);
        }
        
        // Old implementation wich directly operates with the private member-variables of the CGraphPairwise class
/*        for (ptr_node_t &node : getGraphPairwise().m_vNodes)
            node->Pot.setTo(0);
		setState(state, 0);
		for(float &p : P) {
			for (ptr_node_t &node: getGraphPairwise().m_vNodes)
				node->Pot.at<float>(state[node->id], 0) += p / Z;
			incState(state);
		}*/
	}
}
