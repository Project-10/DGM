#include "InferTRW.h"
#include "Graph.h"
#include "macroses.h"

namespace DirectGraphicalModels
{

void CInferTRW::infer(unsigned int nIt)
{
	byte nStates = m_pGraph->m_nStates;					// number of states (classes)

	// ====================================== Initialization ======================================			
	createMessages();
#ifdef USE_PPL
	concurrency::parallel_for_each(m_pGraph->m_vEdges.begin(), m_pGraph->m_vEdges.end(), [nStates](Edge &edge) {
#else
	std::for_each(m_pGraph->m_vEdges.begin(), m_pGraph->m_vEdges.end(), [nStates](Edge &edge) {
#endif
		std::fill(edge.msg, edge.msg + nStates, 1.0f);
		std::fill(edge.msg_temp, edge.msg_temp + nStates, 1.0f);
	});

	// =================================== Calculating messages ==================================	

	calculateMessages(nIt);

	// =================================== Calculating beliefs ===================================	

	for (Node &node : m_pGraph->m_vNodes) {

		size_t nFromEdges = node.from.size();
		for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
			Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
			if (edge_from->node1 < edge_from->node2) {
				Node &src = m_pGraph->m_vNodes[edge_from->node1];
				for (byte s = 0; s < nStates; s++)
					node.Pot.at<float>(s, 0) *= edge_from->Pot.at<float>(s, src.sol);
			}
		}

		// add forward edges
		size_t nToEdges = node.to.size();
		for (size_t e_t = 0; e_t < nToEdges; e_t++) {
			Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
			if (edge_to->node1 < edge_to->node2) {
				for (byte s = 0; s < nStates; s++)
					node.Pot.at<float>(s, 0) *= edge_to->msg[s];
			}
		}

		Point extremumLoc;
		minMaxLoc(node.Pot, NULL, NULL, NULL, &extremumLoc);
		node.sol = static_cast<byte> (extremumLoc.y);
	}

	deleteMessages();
}

void CInferTRW::calculateMessages(unsigned int nIt)
{
	const byte		nStates = m_pGraph->m_nStates;											// number of states
		
	float	*Di = new float[nStates];
	float	*temp = new float[nStates];

	// main loop
	for (unsigned int i = 0; i < nIt; i++) {												// iterations
//#ifdef PRINT_DEBUG_INFO
		if (i == 0) printf("\n");
		if (i % 5 == 0) printf("--- It: %d ---\n", i);
//#endif
		// Forward pass
		std::for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [&](Node &node) {									

			for (byte s = 0; s < nStates; s++) Di[s] = node.Pot.at<float>(s, 0);			// Di = node.pot

			int		nForward = 0;
			size_t	nToEdges = node.to.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					for (byte s = 0; s < nStates; s++) Di[s] *= edge_to->msg[s];			// Di = node.pot * edge_to.msg
					nForward++;
				}
			} // e_t
			
			int		nBackward = 0;
			size_t  nFromEdges = node.from.size();
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					for (byte s = 0; s < nStates; s++) Di[s] *= edge_from->msg[s];			// Di = node.pot * edge_to.msg * edge_from.msg
					nBackward++;
				}
			} // e_f

			for (byte s = 0; s < nStates; s++) Di[s] = powf(Di[s], 1.0f / MAX(nForward, nBackward));

			// pass messages from i to nodes with higher m_ordering
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					DGM_ASSERT(edge_to->node1 == node.id);
					calculateMessage(edge_to, temp, Di, 0);
				}
			} // e_t
		}); 

		// Backward pass
		std::for_each(m_pGraph->m_vNodes.rbegin(), m_pGraph->m_vNodes.rend(), [&](Node &node) {										
			for (byte s = 0; s < nStates; s++) Di[s] = node.Pot.at<float>(s, 0);			// Di = node.pot

			int		nForward = 0;
			int		nBackward = 0;
			size_t	nToEdges = node.to.size();
			size_t  nFromEdges = node.from.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					for (byte s = 0; s < nStates; s++) Di[s] *= edge_to->msg[s];
					nForward++;
				}
			}
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					for (byte s = 0; s < nStates; s++) Di[s] *= edge_from->msg[s];
					nBackward++;
				}
			}

			// normalize Di
			float max = Di[0];
			for (byte s = 1; s < nStates; s++) if (max < Di[s]) max = Di[s];
			for (byte s = 0; s < nStates; s++) Di[s] /= max;
			for (byte s = 0; s < nStates; s++) Di[s] = powf(Di[s], 1.0f / MAX(nForward, nBackward));

			// pass messages from i to nodes with smaller m_ordering
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					DGM_ASSERT(edge_from->node2 == node.id);
					calculateMessage(edge_from, temp, Di, 1);
				}
			}
		}); // All Nodes

	} // iterations

	delete[] Di;
	delete[] temp;
}

// Updates edge->msg
void CInferTRW::calculateMessage(Edge *edge, float *temp, float *source, int dir)
{
	const byte nStates = m_pGraph->m_nStates;

	for (byte s = 0; s < nStates; s++) temp[s] = source[s] / MAX(FLT_EPSILON, edge->msg[s]); 						// tmp = gamma * source - edge.msg

	if (dir == 0) {
		for (byte kdest = 0; kdest < nStates; kdest++) {
			float max = temp[0] * edge->Pot.at<float>(kdest, 0);												// vMin = tmp + edge.Pot(0, kdest)
			for (byte ksource = 1; ksource < nStates; ksource++) {
				float val = temp[ksource] * edge->Pot.at<float>(kdest, ksource);
				if (max < val) max = val;
			}
			edge->msg[kdest] = max;
		}
	} else {
		for (byte kdest = 0; kdest < nStates; kdest++) {
			float max = temp[0] * edge->Pot.at<float>(0, kdest);
			for (byte ksource = 1; ksource < nStates; ksource++) {
				float val = temp[ksource] * edge->Pot.at<float>(ksource, kdest);
				if (max < val) max = val;
			}
			edge->msg[kdest] = max;
		}
	}

	// Normalization
	float max = edge->msg[0];
	for (byte s = 1; s < nStates; s++) if (max < edge->msg[s]) max = edge->msg[s];
	for (byte s = 0; s < nStates; s++) edge->msg[s] /= max;
}
}