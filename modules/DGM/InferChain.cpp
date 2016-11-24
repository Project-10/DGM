#include "InferChain.h"
#include "Graph.h"

namespace DirectGraphicalModels
{
void CInferChain::calculateMessages(unsigned int)
{
	float *temp = new float[m_pGraph->m_nStates];

	// Forward pass
	std::for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end() - 1, [&](ptr_node_t &node) {
		size_t nToEdges = node->to.size();
		for (size_t e_t = 0; e_t < nToEdges; e_t++) {				// outgoing edges
			Edge *edge_to = m_pGraph->m_vEdges[node->to[e_t]].get();	// current outgoing edge
			if (edge_to->node2 == node->id + 1) 
				calculateMessage(edge_to, temp, edge_to->msg);
		} // e_t;
	});

	// Backward pass
	std::for_each(m_pGraph->m_vNodes.rbegin(), m_pGraph->m_vNodes.rend() - 1, [&](ptr_node_t &node) {
		size_t nToEdges = node->to.size();
		for (size_t e_t = 0; e_t < nToEdges; e_t++) {				// outgoing edges
			Edge *edge_to = m_pGraph->m_vEdges[node->to[e_t]].get();	// current outgoing edge
			if (edge_to->node2 == node->id - 1) 
				calculateMessage(edge_to, temp, edge_to->msg);
		} // e_t;
	});

	delete[] temp;
}
}