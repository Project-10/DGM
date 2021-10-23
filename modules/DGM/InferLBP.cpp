#include "InferLBP.h"
#include "GraphPairwise.h"

namespace DirectGraphicalModels
{
	void CInferLBP::calculateMessages(unsigned int nIt)
	{
		const byte		nStates = getGraph().getNumStates();				// number of states
		
		// ======================== Main loop (iterative messages calculation) ========================
		for (unsigned int i = 0; i < nIt; i++) {								// iterations
#ifdef DEBUG_PRINT_INFO
			if (i == 0) printf("\n");
			if (i % 5 == 0) printf("--- It: %d ---\n", i);
#endif
#ifdef ENABLE_PDP
			parallel_for_(Range(0, static_cast<int>(getGraphPairwise().m_vNodes.size())), [&, nStates](const Range& range) {		// all nodes
#else
			const Range range(0, getGraphPairwise().m_vNodes.size());
#endif
			float* temp = new float[nStates];
			for (int i = range.start; i < range.end; i++) {
				ptr_node_t& node = getGraphPairwise().m_vNodes[i];
				// Calculate a message to each neighbor
				for (size_t e_t : node->to) {									// outgoing edges
					Edge* edge_to = getGraphPairwise().m_vEdges[e_t].get();		// current outgoing edge
					calculateMessage(*edge_to, temp, getMessageTemp(e_t), m_maxSum);
				} // e_t;
			} // i
			delete[] temp;
#ifdef ENABLE_PDP			
			});
#endif
			swapMessages();														// Coping data from msg_temp to msg
		} // iterations
	}
}
