#include "InferTree.h"
#include "GraphPairwise.h"

namespace DirectGraphicalModels
{
void CInferTree::calculateMessages(unsigned int)
{
	const byte		nStates	= getGraph().getNumStates();
	const size_t	nNodes	= getGraph().getNumNodes();
	const size_t	nEdges	= getGraph().getNumEdges();

	// ====================================== Initialization ======================================
	vec_bool_t		isReady(nEdges, false);								// Flags indicating whether the messages were already calculated
	vec_bool_t		suspend(nEdges, false);								// Flags indicating weather the message calculation must be postponed
	
	// =================================== Computing messages ===================================	
	size_t  * nFromEdges = new size_t[nNodes];							// Count number of neighbors
	std::deque<size_t> nodeQueue;
	for (ptr_node_t &node: getGraphPairwise().m_vNodes) {
		nFromEdges[node->id] = node->from.size();						// number of incoming edges
		if (nFromEdges[node->id] <= 1) nodeQueue.push_back(node->id);	// Add all leafs to the queue
	}


	float *temp = new float[nStates];
	while (!nodeQueue.empty()) {
		//for (size_t q = 0; q < nodeQueue.size(); q++) printf("%d, ", nodeQueue[q]);	printf("\n");
			
		size_t n = nodeQueue.front();					// n - node with one neighbour
		nodeQueue.pop_front();

		Node  *node     = getGraphPairwise().m_vNodes[n].get();	// Node with one neighbour
		size_t nToEdges = node->to.size();
			
		bool allSuspend = true;
		for (size_t e_t = 0; e_t < nToEdges; e_t++)
			if (!suspend[node->to[e_t]]) {
				allSuspend = false;
				break;
			}

		if (allSuspend) {	// Now prepare messages for suspending edges
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = getGraphPairwise().m_vEdges[node->to[e_t]].get();
				if (isReady[node->to[e_t]]) continue;
				float *msg = &m_msg[node->to[e_t] * nStates];
				calculateMessage(edge_to, temp, msg);
				isReady[node->to[e_t]] = true;
				
				// ------
				size_t n1 = edge_to->node1;
				size_t n2 = edge_to->node2;
				auto it = std::find_if(getGraphPairwise().m_vNodes[n1]->from.begin(), getGraphPairwise().m_vNodes[n1]->from.end(), [&](size_t e) {
					return (getGraphPairwise().m_vEdges[e]->node1 == n2);
				});
				if (it != getGraphPairwise().m_vNodes[n1]->from.end())
					suspend[*it] = true;
				// ------
				
				nFromEdges[n2]--;
				if (nFromEdges[n2] <= 1) nodeQueue.push_back(n2);
			}
		} else {			// Prepare messages for all non-suspending edges	
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge * edge_to = getGraphPairwise().m_vEdges[node->to[e_t]].get();
				if (suspend[node->to[e_t]]) continue;
				if (isReady[node->to[e_t]]) continue;
				
				float *msg = &m_msg[node->to[e_t] * nStates];
				calculateMessage(edge_to, temp, msg);
				isReady[node->to[e_t]] = true;
				// ------
				size_t n1 = edge_to->node1;
				size_t n2 = edge_to->node2;
				auto it = std::find_if(getGraphPairwise().m_vNodes[n1]->from.begin(), getGraphPairwise().m_vNodes[n1]->from.end(), [&](size_t e) {
					return (getGraphPairwise().m_vEdges[e]->node1 == n2);
				});
				if (it != getGraphPairwise().m_vNodes[n1]->from.end())
					suspend[*it] = true;
				// ------
				
				nFromEdges[n2]--;
				if (nFromEdges[n2] <= 1) nodeQueue.push_back(n2);
			}
		}
	} // while

	delete[] temp;
	delete[] nFromEdges;
}
}
