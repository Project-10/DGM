#include "InferTree.h"
#include "Graph.h"

namespace DirectGraphicalModels
{
void CInferTree::calculateMessages(unsigned int)
{
	byte	nStates = m_pGraph->m_nStates;
	size_t	nNodes = m_pGraph->getNumNodes();

	// ====================================== Initialization ======================================
	std::for_each(m_pGraph->m_vEdges.begin(), m_pGraph->m_vEdges.end(), [nStates](Edge &edge) {
		delete[] edge.msg;
		edge.msg = NULL;
		edge.suspend = false;
	});

	// =================================== Computing messages ===================================	
	size_t  * nFromEdges = new size_t[nNodes];					// Count number of neighbors
	std::deque<size_t> nodeQueue;
	std::for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [&](Node &node) {
		nFromEdges[node.id] = node.from.size();							// number of incoming edges
		if (nFromEdges[node.id] <= 1) nodeQueue.push_back(node.id);		// Add all leafs to the queue
	});


	float *temp = new float[nStates];
	while (!nodeQueue.empty()) {
		//for (size_t q = 0; q < nodeQueue.size(); q++) printf("%d, ", nodeQueue[q]);	printf("\n");
			
		size_t n = nodeQueue.front();				// n - node with one neighbour
		nodeQueue.pop_front();

		Node * node     = &m_pGraph->m_vNodes[n];	// Node with one neighbour
		size_t nToEdges = node->to.size();
			
		bool allSuspend = true;
		for (size_t e_t = 0; e_t < nToEdges; e_t++) {
			Edge *edge_to = &m_pGraph->m_vEdges[node->to[e_t]];
			if (!edge_to->suspend) {
				allSuspend = false;
				break;
			}
		}

		if (allSuspend) {	// Now prepare messages for suspending edges
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node->to[e_t]];
				if (edge_to->msg) continue;
					
				calculateMessage(edge_to, temp, edge_to->msg);
					
				size_t n2 = edge_to->node2;
				nFromEdges[n2]--;
				if (nFromEdges[n2] <= 1) nodeQueue.push_back(n2);
			}
		} else {			// Prepare messages for all non-suspending edges	
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge * edge_to = &m_pGraph->m_vEdges[node->to[e_t]];
				if (edge_to->suspend) continue;
				if (edge_to->msg)     continue;
					
				calculateMessage(edge_to, temp, edge_to->msg);
					
				size_t n2 = edge_to->node2;
				nFromEdges[n2]--;
				if (nFromEdges[n2] <= 1) nodeQueue.push_back(n2);
			}
		}

	} // while

	delete[] temp;
	delete[] nFromEdges;
}
}