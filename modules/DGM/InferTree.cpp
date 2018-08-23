#include "InferTree.h"
#include "GraphPairwise.h"

namespace DirectGraphicalModels
{
void CInferTree::calculateMessages(unsigned int)
{
	byte	nStates = getGraph()->getNumStates();
	size_t	nNodes	= getGraph()->getNumNodes();

	// ====================================== Initialization ======================================
	for (ptr_edge_t &edge: getGraphPairwise()->m_vEdges) {
		delete[] edge->msg;
		edge->msg = NULL;
		edge->suspend = false;
	}

	// =================================== Computing messages ===================================	
	size_t  * nFromEdges = new size_t[nNodes];							// Count number of neighbors
	std::deque<size_t> nodeQueue;
	for (ptr_node_t &node: getGraphPairwise()->m_vNodes) {
		nFromEdges[node->id] = node->from.size();						// number of incoming edges
		if (nFromEdges[node->id] <= 1) nodeQueue.push_back(node->id);	// Add all leafs to the queue
	}


	float *temp = new float[nStates];
	while (!nodeQueue.empty()) {
		//for (size_t q = 0; q < nodeQueue.size(); q++) printf("%d, ", nodeQueue[q]);	printf("\n");
			
		size_t n = nodeQueue.front();					// n - node with one neighbour
		nodeQueue.pop_front();

		Node  *node     = getGraphPairwise()->m_vNodes[n].get();	// Node with one neighbour
		size_t nToEdges = node->to.size();
			
		bool allSuspend = true;
		for (size_t e_t = 0; e_t < nToEdges; e_t++) {
			Edge *edge_to = getGraphPairwise()->m_vEdges[node->to[e_t]].get();
			if (!edge_to->suspend) {
				allSuspend = false;
				break;
			}
		}

		if (allSuspend) {	// Now prepare messages for suspending edges
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = getGraphPairwise()->m_vEdges[node->to[e_t]].get();
				if (edge_to->msg) continue;
					
				calculateMessage(edge_to, temp, edge_to->msg);
					
				size_t n2 = edge_to->node2;
				nFromEdges[n2]--;
				if (nFromEdges[n2] <= 1) nodeQueue.push_back(n2);
			}
		} else {			// Prepare messages for all non-suspending edges	
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge * edge_to = getGraphPairwise()->m_vEdges[node->to[e_t]].get();
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