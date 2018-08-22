#include "InferTRW.h"
#include "GraphPairwise.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
void CInferTRW::infer(unsigned int nIt)
{
	const byte nStates = getGraph()->getNumStates();					// number of states (classes)

	// ====================================== Initialization ======================================			
	createMessages();
#ifdef ENABLE_PPL
	concurrency::parallel_for_each(getGraphPairwise()->m_vEdges.begin(), getGraphPairwise()->m_vEdges.end(), [nStates](ptr_edge_t &edge) {
#else
	std::for_each(getGraphPairwise()->m_vEdges.begin(), getGraphPairwise()->m_vEdges.end(), [nStates](ptr_edge_t &edge) {
#endif
		std::fill(edge->msg, edge->msg + nStates, 1.0f);
		std::fill(edge->msg_temp, edge->msg_temp + nStates, 1.0f);
	});

	// =================================== Calculating messages ==================================	

	calculateMessages(nIt);

	// =================================== Calculating beliefs ===================================	

	for (ptr_node_t &node : getGraphPairwise()->m_vNodes) {
		// backward edges
		for (size_t e_f : node->from) {
			Edge * edge_from = getGraphPairwise()->m_vEdges[e_f].get();
			if (edge_from->node1 > edge_from->node2) continue;
			Node *src = getGraphPairwise()->m_vNodes[edge_from->node1].get();
			for (byte s = 0; s < nStates; s++) node->Pot.at<float>(s, 0) *= edge_from->Pot.at<float>(src->sol, s);
		}
		// forward edges
		for (size_t e_t : node->to) {
			Edge *edge_to = getGraphPairwise()->m_vEdges[e_t].get();
			if (edge_to->node1 > edge_to->node2) continue;
			for (byte s = 0; s < nStates; s++) node->Pot.at<float>(s, 0) *= edge_to->msg[s];
		}

		Point extremumLoc;
		minMaxLoc(node->Pot, NULL, NULL, NULL, &extremumLoc);
		node->sol = static_cast<byte> (extremumLoc.y);
	}

	deleteMessages();
}

void CInferTRW::calculateMessages(unsigned int nIt)
{
	const    byte	  nStates	= getGraph()->getNumStates();										// number of states
	float			* data		= new float[nStates];
	float			* temp		= new float[nStates];

	// main loop
	for (unsigned int i = 0; i < nIt; i++) {												// iterations
#ifdef DEBUG_PRINT_INFO
		if (i == 0) printf("\n");
		if (i % 5 == 0) printf("--- It: %d ---\n", i);
#endif
		// Forward pass
		std::for_each(getGraphPairwise()->m_vNodes.begin(), getGraphPairwise()->m_vNodes.end(), [&](ptr_node_t &node) {
			memcpy(data, node->Pot.data, nStates * sizeof(float));							// data = node.pot

			int	nForward = 0;
			for (size_t e_t : node->to) {
				Edge *edge_to = getGraphPairwise()->m_vEdges[e_t].get();
				if (edge_to->node1 > edge_to->node2) continue;
				for (byte s = 0; s < nStates; s++) data[s] *= edge_to->msg[s];				// data = node.pot * edge_to.msg
				nForward++;
			} // e_t
			
			int	nBackward = 0;
			for (size_t e_f : node->from) {
				Edge *edge_from = getGraphPairwise()->m_vEdges[e_f].get();
				if (edge_from->node1 > edge_from->node2) continue;
				for (byte s = 0; s < nStates; s++) data[s] *= edge_from->msg[s];				// data = node.pot * edge_to.msg * edge_from.msg
				nBackward++;
			} // e_f

			for (byte s = 0; s < nStates; s++) data[s] = static_cast<float>(fastPow(data[s], 1.0f / MAX(nForward, nBackward)));

			// pass messages from i to nodes with higher m_ordering
			for (size_t e_t : node->to) {
				Edge *edge_to = getGraphPairwise()->m_vEdges[e_t].get();
				if (edge_to->node1 < edge_to->node2) calculateMessage(*edge_to, temp, data);
			} // e_t
		});

		// Backward pass
		std::for_each(getGraphPairwise()->m_vNodes.rbegin(), getGraphPairwise()->m_vNodes.rend(), [&](ptr_node_t &node) {
			memcpy(data, node->Pot.data, nStates * sizeof(float));							// data = node.pot

			int	nForward = 0;
			for (size_t e_t : node->to) {
				Edge *edge_to = getGraphPairwise()->m_vEdges[e_t].get();
				if (edge_to->node1 > edge_to->node2) continue;
				for (byte s = 0; s < nStates; s++) data[s] *= edge_to->msg[s];
				nForward++;
			} // e_t
			
			int	nBackward = 0;
			for (size_t e_f : node->from) {
				Edge *edge_from = getGraphPairwise()->m_vEdges[e_f].get();
				if (edge_from->node1 > edge_from->node2) continue;
				for (byte s = 0; s < nStates; s++) data[s] *= edge_from->msg[s];
				nBackward++;
			} // e_f

			// normalize data
			float max = data[0];
			for (byte s = 1; s < nStates; s++) if (max < data[s]) max = data[s];
			for (byte s = 0; s < nStates; s++) data[s] /= max;
			for (byte s = 0; s < nStates; s++) data[s] = static_cast<float>(fastPow(data[s], 1.0f / MAX(nForward, nBackward)));

			// pass messages from i to nodes with smaller m_ordering
			for (size_t e_f : node->from) {
				Edge *edge_from = getGraphPairwise()->m_vEdges[e_f].get();
				if (edge_from->node1 < edge_from->node2) calculateMessage(*edge_from, temp, data);
			} // e_f
		}); // All Nodes

	} // iterations

	delete[] data;
	delete[] temp;
}

// Updates edge->msg = F(data, edge.Pot)
void CInferTRW::calculateMessage(Edge &edge, float *temp, float *data)
{
	const byte		nStates = getGraph()->getNumStates();

	for (byte s = 0; s < nStates; s++) temp[s] = data[s] / MAX(FLT_EPSILON, edge.msg[s]); 				// tmp = gamma * data / edge.msg

	for (byte y = 0; y < nStates; y++) {
		float *pPot = edge.Pot.ptr<float>(y);
		float max = temp[0] * pPot[0];																// vMin = tmp + edge.Pot(0, kdest)
		for (byte x = 1; x < nStates; x++) {
			float val = temp[x] * pPot[x];
			if (max < val) max = val;
		}
		edge.msg[y] = max;
	}

	// Normalization
	float max = edge.msg[0];
	for (byte s = 1; s < nStates; s++) if (max < edge.msg[s]) max = edge.msg[s];
	for (byte s = 0; s < nStates; s++) edge.msg[s] /= max;
}
}