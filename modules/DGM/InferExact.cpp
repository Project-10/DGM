#include "InferExact.h"
#include "Graph.h"
#include <numeric>

namespace DirectGraphicalModels
{
void CInferExact::infer(unsigned int)
{
	size_t		nNodes = m_pGraph->getNumNodes();		
	vec_byte_t	state(nNodes);				
	
	// Calculating the potentials for every possible configuration
	vec_float_t	P = calculatePotentials();

	// Calculating the partition function
	float Z = std::accumulate(P.cbegin(), P.cend(), 0.0f);
	
	// Filling node potentials with marginal probabilities
	for (ptr_node_t &node : m_pGraph->m_vNodes) node->Pot.setTo(0);
	setState(state, 0);
	for(float &p : P) {
		for (ptr_node_t &node: m_pGraph->m_vNodes) 
			node->Pot.at<float>(state[node->id], 0) += p / Z;
		incState(state);
	};
}
}
