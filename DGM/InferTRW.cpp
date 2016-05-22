#include "InferTRW.h"
#include "Graph.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
byte FindMin(float *data, byte len)
{
	byte res = 0;
	float minVal = data[0];
	for (byte i = 1; i < len; i++)
		if (minVal > data[i]) {
			minVal = data[i];
			res = i;
		}
	return res;
}

void ComputeAndSubtractMin(float *data, byte len)
{
	float min = data[0];
	for (byte s = 1; s < len; s++) if (min > data[s]) min = data[s];
	for (byte s = 0; s < len; s++) data[s] -= min;
}

void CInferTRW::infer(unsigned int nIt)
{
	byte nStates = m_pGraph->m_nStates;					// number of states (classes)

	// ====================================== Initialization ======================================			
	createMessages();
	for (Edge &edge : m_pGraph->m_vEdges) {
		std::fill(edge.msg, edge.msg + nStates, 0.0f);
		std::fill(edge.msg_temp, edge.msg_temp + nStates, 0.0f);
	}
	transformPotentials();

	// =================================== Calculating messages ==================================	
	calculateMessages(nIt);

	// =================================== Calculating beliefs ===================================	
	for (Node &node : m_pGraph->m_vNodes) {

		size_t nToEdges = node.to.size();
		size_t nFromEdges = node.from.size();
		for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
			Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
			if (edge_from->node1 < edge_from->node2) {
				Node &src = m_pGraph->m_vNodes[edge_from->node1];
				for (byte s = 0; s < nStates; s++)
					node.Pot.at<float>(s, 0) += edge_from->Pot.at<float>(s, src.sol);
			}
		}

		// add forward edges
		for (size_t e_t = 0; e_t < nToEdges; e_t++) {
			Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
			if (edge_to->node1 < edge_to->node2) {
				for (byte s = 0; s < nStates; s++)
					node.Pot.at<float>(s, 0) += edge_to->msg[s];
			}
		}

		node.sol = FindMin((float *) node.Pot.data, nStates);
	}


	transformPotentialsBack();
	deleteMessages();
}

void CInferTRW::calculateMessages(unsigned int nIt)
{
	const byte		nStates = m_pGraph->m_nStates;											// number of states
		
	float	*Di = new float[nStates];
	float	*temp = new float[nStates];

	// main loop
	for (unsigned int i = 0; i < nIt; i++) {												// iterations
#ifdef PRINT_DEBUG_INFO
		if (i == 0) printf("\n");
		printf("--- It: %d ---\n", i);
#endif
		// Forward pass
		std::for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [&](Node &node) {									
			memcpy(Di, (float *)node.Pot.data, nStates * sizeof(float));	// Di = node.pot

			int		nForward = 0;
			int		nBackward = 0;
			size_t	nToEdges = node.to.size();
			size_t  nFromEdges = node.from.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					for (byte s = 0; s < nStates; s++) Di[s] += edge_to->msg[s];
					nForward++;
				}
			} // e_t
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					for (byte s = 0; s < nStates; s++) Di[s] += edge_from->msg[s];
					nBackward++;
				}
			} // e_f

			// pass messages from i to nodes with higher m_ordering
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					DGM_ASSERT(edge_to->node1 == node.id);
					calculateMessage(edge_to, temp, Di, 1.0f / MAX(nForward, nBackward), 0);
				}
			} // e_t
		}); 

		// Backward pass
		std::for_each(m_pGraph->m_vNodes.rbegin(), m_pGraph->m_vNodes.rend(), [&](Node &node) {										
			memcpy(Di, (float *)node.Pot.data, nStates * sizeof(float));

			int		nForward = 0;
			int		nBackward = 0;
			size_t	nToEdges = node.to.size();
			size_t  nFromEdges = node.from.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					for (byte s = 0; s < nStates; s++) Di[s] += edge_to->msg[s];
					nForward++;
				}
			}
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					for (byte s = 0; s < nStates; s++) Di[s] += edge_from->msg[s];
					nBackward++;
				}
			}

			// normalize Di
			ComputeAndSubtractMin(Di, nStates);

			// pass messages from i to nodes with smaller m_ordering
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					DGM_ASSERT(edge_from->node2 == node.id);
					calculateMessage(edge_from, temp, Di, 1.0f / MAX(nForward, nBackward), 1);
				}
			}
		}); // All Nodes

	} // iterations

	delete[] Di;
	delete[] temp;
}

// ================================ PRIVATE ================================

void CInferTRW::transformPotentials()
{
	const byte nStates = m_pGraph->m_nStates;

	for (Node &node : m_pGraph->m_vNodes) {
		for (byte j = 0; j < nStates; j++)
			node.Pot.at<float>(j, 0) = -logf(MAX(FLT_EPSILON, node.Pot.at<float>(j, 0)));
	}

	for (Edge &edge : m_pGraph->m_vEdges) {
		for (byte j = 0; j < nStates; j++)
			for (byte i = 0; i < nStates; i++)
				edge.Pot.at<float>(j, i) = -logf(MAX(FLT_EPSILON, edge.Pot.at<float>(j, i)));
	}
}

void CInferTRW::transformPotentialsBack()
{
	const byte nStates = m_pGraph->m_nStates;

	for (Node &node : m_pGraph->m_vNodes) {
		for (byte j = 0; j < nStates; j++)
			node.Pot.at<float>(j, 0) = expf(-node.Pot.at<float>(j, 0));
	}

	for (Edge &edge : m_pGraph->m_vEdges) {
		for (byte j = 0; j < nStates; j++)
			for (byte i = 0; i < nStates; i++)
				edge.Pot.at<float>(j, i) = expf(-edge.Pot.at<float>(j, i));
	}
}

// When UpdateMessage() is called, edge contains message from dest to source.
// The function must replace it with the message from source to dest.
// The update rule is given below assuming that source corresponds to tail (i) and dest corresponds
// to head (j) (which is the case if dir==0).
//
// 1. Compute Di[ki] = gamma*source[ki] - message[ki].  (Note: message = message from j to i).
// 2. Compute distance transform: set
//       message[kj] = min_{ki} (Di[ki] + V(ki,kj)). (Note: message = message from i to j).
// 3. Compute vMin = min_{kj} m_message[kj].
// 4. Set m_message[kj] -= vMin.
// 5. Return vMin.
//
// If dir==1 then source corresponds to j, sink corresponds to i. Then the update rule is
//
// 1. Compute Dj[kj] = gamma*source[kj] - message[kj].  (Note: message = message from i to j).
// 2. Compute distance transform: set
//       message[ki] = min_{kj} (Dj[kj] + V(ki,kj)). (Note: message = message from j to i).
// 3. Compute vMin = min_{ki} m_message[ki].
// 4. Set m_message[ki] -= vMin.
// 5. Return vMin.
	
// Updates edge->msg
void CInferTRW::calculateMessage(Edge *edge, float *temp, float *source, float gamma, int dir)
{
	const byte nStates = m_pGraph->m_nStates;

	for (byte s = 0; s < nStates; s++) temp[s] = gamma * source[s] - edge->msg[s];		// tmp = gamma * source - edge.msg

	if (dir == 0) {
		for (byte kdest = 0; kdest < nStates; kdest++) {
			float vMin = temp[0] + edge->Pot.at<float>(kdest, 0);									// vMin = tmp + edge.Pot(0, kdest)
			for (byte ksource = 1; ksource < nStates; ksource++) {
				if (vMin > temp[ksource] + edge->Pot.at<float>(kdest, ksource))
					vMin = temp[ksource] + edge->Pot.at<float>(kdest, ksource);
			}
			edge->msg[kdest] = vMin;
		}
	}
	else {
		for (int kdest = 0; kdest < nStates; kdest++) {
			float vMin = temp[0] + edge->Pot.at<float>(0, kdest);
			for (int ksource = 1; ksource < nStates; ksource++) {
				if (vMin > temp[ksource] + edge->Pot.at<float>(ksource, kdest))
					vMin = temp[ksource] + edge->Pot.at<float>(ksource, kdest);
			}
			edge->msg[kdest] = vMin;
		}
	}

	ComputeAndSubtractMin(edge->msg, nStates);
}
}