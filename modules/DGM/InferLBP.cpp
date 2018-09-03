#include "InferLBP.h"
#include "GraphPairwise.h"

namespace DirectGraphicalModels
{
void CInferLBP::calculateMessages(unsigned int nIt)
{
	const byte		nStates = getGraph().getNumStates();				// number of states
	
	// ======================== Main loop (iterative messages calculation) ========================
#ifndef ENABLE_PPL
	float *temp = new float[nStates];
#endif
	for (unsigned int i = 0; i < nIt; i++) {					// iterations
#ifdef DEBUG_PRINT_INFO
		if (i == 0) printf("\n");
		if (i % 5 == 0) printf("--- It: %d ---\n", i);
#endif
#ifdef ENABLE_PPL		
		concurrency::parallel_for_each(getGraphPairwise().m_vNodes.begin(), getGraphPairwise().m_vNodes.end(), [&, nStates](ptr_node_t &node) {		// all nodes
			float *temp = new float[nStates];
#else
		std::for_each(getGraphPairwise().m_vNodes.begin(), getGraphPairwise().m_vNodes.end(), [&](ptr_node_t &node) {
#endif
			// Calculate a message to each neighbor
			size_t nToEdges = node->to.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {				// outgoing edges
				Edge *edge_to = getGraphPairwise().m_vEdges[node->to[e_t]].get();	// current outgoing edge
				calculateMessage(edge_to, temp, edge_to->msg_temp, m_maxSum);
			} // e_t;
#ifdef ENABLE_PPL
			delete[] temp;
#endif
		}); // nodes
		swapMessages();										// Coping data from msg_temp to msg
	} // iterations
#ifndef ENABLE_PPL
	delete[] temp;
#endif
}
}