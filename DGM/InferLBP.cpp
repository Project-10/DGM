#include "InferLBP.h"
#include "Graph.h"

namespace DirectGraphicalModels
{
void CInferLBP::calculateMessages(unsigned int nIt)
{
	const size_t	nNodes	= m_pGraph->getNumNodes();			// number of nodes
	const byte		nStates = m_pGraph->m_nStates;				// number of states
	
	// ======================== Main loop (iterative messages calculation) ========================
#ifndef USE_PPL
	float *temp = new float[nStates];
#endif
	for (unsigned int i = 0; i < nIt; i++) {					// iterations
#ifdef PRINT_DEBUG_INFO
		if (i == 0) printf("\n");
		printf("--- It: %d ---\n", i);
#endif
#ifdef USE_PPL		
		concurrency::parallel_for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [&, nStates](Node &node) {		// all nodes
			float *temp = new float[nStates];
#else
		std::for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [&](Node &node) {
#endif
			// Calculate a message to each neighbor
			size_t nToEdges = node.to.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {				// outgoing edges
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];		// current outgoing edge
				calculateMessage(edge_to, temp, edge_to->msg_temp, m_maxSum);
			} // e_t;
#ifdef USE_PPL
			delete[] temp;
#endif
		}); // nodes
		swapMessages();										// Coping data from msg_temp to msg

	} // iterations
#ifndef USE_PPL
	delete[] temp;
#endif

}
}