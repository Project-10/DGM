#include "InferChain.h"
#include "GraphPairwise.h"

namespace DirectGraphicalModels
{
	void CInferChain::calculateMessages(unsigned int)
	{
		float *temp = new float[getGraph().getNumNodes()];

		// Forward pass
		std::for_each(getGraphPairwise().m_vNodes.begin(), getGraphPairwise().m_vNodes.end() - 1, [&](ptr_node_t &node) {
			for (size_t e_t : node->to) {									// outgoing edges
				ptr_edge_t &edge_to = getGraphPairwise().m_vEdges[e_t];			// current outgoing edge
				if (edge_to->node2 == node->id + 1)
					calculateMessage(*edge_to, temp, getMessage(e_t));
			} // e_t;
		});

		// Backward pass
		std::for_each(getGraphPairwise().m_vNodes.rbegin(), getGraphPairwise().m_vNodes.rend() - 1, [&](ptr_node_t &node) {
			for (size_t e_t : node->to) {									// outgoing edges
				Edge *edge_to = getGraphPairwise().m_vEdges[e_t].get();		// current outgoing edge
				if (edge_to->node2 == node->id - 1)
					calculateMessage(*edge_to, temp, getMessage(e_t));
			} // e_t;
		});

		delete[] temp;
	}
}
