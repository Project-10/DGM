#include "InferChain.h"
#include "GraphPairwise.h"

namespace DirectGraphicalModels
{
	void CInferChain::calculateMessages(unsigned int)
	{
		const byte nStates = getGraph().getNumStates();
		float *temp = new float[getGraph().getNumNodes()];

		// Forward pass
		std::for_each(getGraphPairwise().m_vNodes.begin(), getGraphPairwise().m_vNodes.end() - 1, [&](ptr_node_t &node) {
			for (size_t e_t : node->to) {									// outgoing edges
				Edge *edge_to = getGraphPairwise().m_vEdges[e_t].get();		// current outgoing edge
				float *msg = &m_msg[e_t * nStates];							// message of current outgoing edge
				if (edge_to->node2 == node->id + 1)
					calculateMessage(*edge_to, temp, msg);
			} // e_t;
		});

		// Backward pass
		std::for_each(getGraphPairwise().m_vNodes.rbegin(), getGraphPairwise().m_vNodes.rend() - 1, [&](ptr_node_t &node) {
			for (size_t e_t : node->to) {									// outgoing edges
				Edge *edge_to = getGraphPairwise().m_vEdges[e_t].get();		// current outgoing edge
				float *msg = &m_msg[e_t * nStates];							// message of current outgoing edge
				if (edge_to->node2 == node->id - 1)
					calculateMessage(*edge_to, temp, msg);
			} // e_t;
		});

		delete[] temp;
	}
}
