#include "InferTRW.h"
#include "Graph.h"
#include "macroses.h"

namespace DirectGraphicalModels
{

void CInferTRW::infer(unsigned int nIt)
{
	infer_log(nIt);
	return;
	
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


void CInferTRW::infer_log(unsigned int nIt)
{
	byte nStates = m_pGraph->m_nStates;					// number of states (classes)

														// ====================================== Initialization ======================================			
	createMessages();
#ifdef USE_PPL
	concurrency::parallel_for_each(m_pGraph->m_vEdges.begin(), m_pGraph->m_vEdges.end(), [nStates](Edge &edge) {
#else
	std::for_each(m_pGraph->m_vEdges.begin(), m_pGraph->m_vEdges.end(), [nStates](Edge &edge) {
#endif
		std::fill(edge.msg, edge.msg + nStates, 0.0f);
		std::fill(edge.msg_temp, edge.msg_temp + nStates, 0.0f);
	});

	// =================================== Calculating messages ==================================	
	transformLog();
	calculateMessages_log(nIt);

	// =================================== Calculating beliefs ===================================	

	for (Node &node : m_pGraph->m_vNodes) {

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
		size_t nToEdges = node.to.size();
		for (size_t e_t = 0; e_t < nToEdges; e_t++) {
			Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
			if (edge_to->node1 < edge_to->node2) {
				for (byte s = 0; s < nStates; s++)
					node.Pot.at<float>(s, 0) += edge_to->msg[s];
			}
		}

		Point extremumLoc;
		minMaxLoc(node.Pot, NULL, NULL, &extremumLoc, NULL);
		node.sol = static_cast<byte> (extremumLoc.y);
	}

	transformExp();
	deleteMessages();

}


void CInferTRW::calculateMessages(unsigned int nIt)
{
	register byte	  s;																	// state indexes
	const    byte	  nStates	= m_pGraph->m_nStates;										// number of states
	float			* data		= new float[nStates];
	float			* temp		= new float[nStates];

	// main loop
	for (unsigned int i = 0; i < nIt; i++) {												// iterations
//#ifdef PRINT_DEBUG_INFO
		if (i == 0) printf("\n");
		if (i % 5 == 0) printf("--- It: %d ---\n", i);
//#endif
		// Forward pass
		std::for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [&](Node &node) {									

			for (s = 0; s < nStates; s++) data[s] = node.Pot.at<float>(s, 0);				// data = node.pot

			int		nForward = 0;
			size_t	nToEdges = node.to.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					for (byte s = 0; s < nStates; s++) data[s] *= edge_to->msg[s];			// data = node.pot * edge_to.msg
					nForward++;
				}
			} // e_t
			
			int		nBackward = 0;
			size_t  nFromEdges = node.from.size();
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					for (s = 0; s < nStates; s++) data[s] *= edge_from->msg[s];			// data = node.pot * edge_to.msg * edge_from.msg
					nBackward++;
				}
			} // e_f

			for (s = 0; s < nStates; s++) data[s] = powf(data[s], 1.0f / MAX(nForward, nBackward));

			// pass messages from i to nodes with higher m_ordering
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					DGM_ASSERT(edge_to->node1 == node.id);
					calculateMessage(edge_to, temp, data, 0);
				}
			} // e_t
		}); 

		// Backward pass
		std::for_each(m_pGraph->m_vNodes.rbegin(), m_pGraph->m_vNodes.rend(), [&](Node &node) {										
			for (byte s = 0; s < nStates; s++) data[s] = node.Pot.at<float>(s, 0);			// data = node.pot

			int		nForward = 0;
			size_t	nToEdges = node.to.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					for (s = 0; s < nStates; s++) data[s] *= edge_to->msg[s];
					nForward++;
				}
			} // e_t
			
			int		nBackward = 0;
			size_t  nFromEdges = node.from.size();
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					for (s = 0; s < nStates; s++) data[s] *= edge_from->msg[s];
					nBackward++;
				}
			} // e_f

			// normalize Di
			float max = data[0];
			for (s = 1; s < nStates; s++) if (max < data[s]) max = data[s];
			for (s = 0; s < nStates; s++) data[s] /= max;
			for (s = 0; s < nStates; s++) data[s] = powf(data[s], 1.0f / MAX(nForward, nBackward));

			// pass messages from i to nodes with smaller m_ordering
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					DGM_ASSERT(edge_from->node2 == node.id);
					calculateMessage(edge_from, temp, data, 1);
				}
			} // e_f
		}); // All Nodes

	} // iterations

	delete[] data;
	delete[] temp;
}

void CInferTRW::calculateMessages_log(unsigned int nIt)
{
	register byte	  s;																	// state indexes
	const    byte	  nStates = m_pGraph->m_nStates;										// number of states
	float			* data = new float[nStates];
	float			* temp = new float[nStates];

	// main loop
	for (unsigned int i = 0; i < nIt; i++) {												// iterations
																							//#ifdef PRINT_DEBUG_INFO
		if (i == 0) printf("\n");
		if (i % 5 == 0) printf("--- It: %d ---\n", i);
		//#endif
		// Forward pass
		for (Node &node: m_pGraph->m_vNodes) {

			for (s = 0; s < nStates; s++) data[s] = node.Pot.at<float>(s, 0);				// data = node.pot

			int		nForward = 0;
			size_t	nToEdges = node.to.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					for (s = 0; s < nStates; s++) data[s] += edge_to->msg[s];				// data = node.pot * edge_to.msg
					nForward++;
				}
			} // e_t

			int		nBackward = 0;
			size_t  nFromEdges = node.from.size();
			for (size_t e_f : node.from) {
				Edge &edge_from = m_pGraph->m_vEdges[e_f];
				if (edge_from.node1 < edge_from.node2) {
					for (s = 0; s < nStates; s++) data[s] += edge_from.msg[s];			// data = node.pot * edge_to.msg * edge_from.msg
					nBackward++;
				}
			}
			//for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
			//	Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
			//	if (edge_from->node1 < edge_from->node2) {
			//		for (s = 0; s < nStates; s++) data[s] += edge_from->msg[s];			// data = node.pot * edge_to.msg * edge_from.msg
			//		nBackward++;
			//	}
			//} // e_f

			float gamma = 1.0f / MAX(nForward, nBackward);
			for (s = 0; s < nStates; s++) data[s] *= gamma;

			// pass messages from i to nodes with higher m_ordering
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					//DGM_ASSERT(edge_to->node1 == node.id);
					calculateMessage_log(edge_to, temp, data, 0);
				}
			} // e_t
		};

		// Backward pass
		std::for_each(m_pGraph->m_vNodes.rbegin(), m_pGraph->m_vNodes.rend(), [&](Node &node) {
			for (byte s = 0; s < nStates; s++) data[s] = node.Pot.at<float>(s, 0);			// data = node.pot

			int		nForward = 0;
			size_t	nToEdges = node.to.size();
			for (size_t e_t = 0; e_t < nToEdges; e_t++) {
				Edge *edge_to = &m_pGraph->m_vEdges[node.to[e_t]];
				if (edge_to->node1 < edge_to->node2) {
					for (s = 0; s < nStates; s++) data[s] += edge_to->msg[s];
					nForward++;
				}
			} // e_t

			int		nBackward = 0;
			size_t  nFromEdges = node.from.size();
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					for (s = 0; s < nStates; s++) data[s] += edge_from->msg[s];
					nBackward++;
				}
			} // e_f

			  // normalize Di
			float min = data[0];
			for (s = 1; s < nStates; s++) if (min > data[s]) min = data[s];
			for (s = 0; s < nStates; s++) data[s] -= min;
			for (s = 0; s < nStates; s++) data[s] = data[s] / MAX(nForward, nBackward);

			// pass messages from i to nodes with smaller m_ordering
			for (size_t e_f = 0; e_f < nFromEdges; e_f++) {
				Edge *edge_from = &m_pGraph->m_vEdges[node.from[e_f]];
				if (edge_from->node1 < edge_from->node2) {
					DGM_ASSERT(edge_from->node2 == node.id);
					calculateMessage_log(edge_from, temp, data, 1);
				}
			} // e_f
		}); // All Nodes

	} // iterations

	delete[] data;
	delete[] temp;
}



// Updates edge->msg = F(data, edge.Pot)
void CInferTRW::calculateMessage(Edge *edge, float *temp, float *data, int dir)
{
	register byte	s;																				// state indexes
	const byte		nStates = m_pGraph->m_nStates;

	for (s = 0; s < nStates; s++) temp[s] = data[s] / MAX(FLT_EPSILON, edge->msg[s]); 				// tmp = gamma * data / edge.msg

	if (dir == 0) {
		for (byte kdest = 0; kdest < nStates; kdest++) {
			float max = temp[0] * edge->Pot.at<float>(kdest, 0);									// vMin = tmp + edge.Pot(0, kdest)
			for (byte ksource = 1; ksource < nStates; ksource++) {
				float val = temp[ksource] * edge->Pot.at<float>(kdest, ksource);
				if (max < val) max = val;
			}
			edge->msg[kdest] = max;
		}
	} else {	// TODO: Maybe this is redundant
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
	for (s = 1; s < nStates; s++) if (max < edge->msg[s]) max = edge->msg[s];
	for (s = 0; s < nStates; s++) edge->msg[s] /= max;
}

void CInferTRW::calculateMessage_log(Edge *edge, float *temp, float *data, int dir)
{
	register byte	s;																				// state indexes
	const byte		nStates = m_pGraph->m_nStates;

	for (s = 0; s < nStates; s++) temp[s] = data[s] - edge->msg[s]; 								// tmp = gamma * data - edge.msg

	if (dir == 0) {
		for (byte kdest = 0; kdest < nStates; kdest++) {
			float min = temp[0] + edge->Pot.at<float>(kdest, 0);									// vMin = tmp + edge.Pot(0, kdest)
			for (byte ksource = 1; ksource < nStates; ksource++) {
				float val = temp[ksource] + edge->Pot.at<float>(kdest, ksource);
				if (min > val) min = val;
			}
			edge->msg[kdest] = min;
		}
	}
	else {	// TODO: Maybe this is redundant
		for (byte kdest = 0; kdest < nStates; kdest++) {
			float min = temp[0] + edge->Pot.at<float>(0, kdest);
			for (byte ksource = 1; ksource < nStates; ksource++) {
				float val = temp[ksource] + edge->Pot.at<float>(ksource, kdest);
				if (min > val) min = val;
			}
			edge->msg[kdest] = min;
		}
	}

	// Normalization
	float min = edge->msg[0];
	for (s = 1; s < nStates; s++) if (min > edge->msg[s]) min = edge->msg[s];
	for (s = 0; s < nStates; s++) edge->msg[s] -= min;
}





void CInferTRW::transformLog(void)
{
//#ifdef USE_PPL
//	concurrency::parallel_for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [](Node &node) { log(node.Pot, node.Pot); node.Pot *= -1.0f; });
//	concurrency::parallel_for_each(m_pGraph->m_vEdges.begin(), m_pGraph->m_vEdges.end(), [](Edge &edge) { log(edge.Pot, edge.Pot); edge.Pot *= -1.0f; });
//#else
//	std::for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [](Node &node) { log(node.Pot, node.Pot); node.Pot *= -1.0f; });
//	std::for_each(m_pGraph->m_vEdges.begin(), m_pGraph->m_vEdges.end(), [](Edge &edge) { log(edge.Pot, edge.Pot); edge.Pot *= -1.0f; });
//#endif	

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

void CInferTRW::transformExp(void)
{
#ifdef USE_PPL
	concurrency::parallel_for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [](Node &node) { exp(-node.Pot, node.Pot); });
	concurrency::parallel_for_each(m_pGraph->m_vEdges.begin(), m_pGraph->m_vEdges.end(), [](Edge &edge) { exp(-edge.Pot, edge.Pot); });
#else
	std::for_each(m_pGraph->m_vNodes.begin(), m_pGraph->m_vNodes.end(), [](Node &node) { exp(-node.Pot, node.Pot); });
	std::for_each(m_pGraph->m_vEdges.begin(), m_pGraph->m_vEdges.end(), [](Edge &edge) { exp(-edge.Pot, edge.Pot); });
#endif	
	//const byte nStates = m_pGraph->m_nStates;

	//for (Node &node : m_pGraph->m_vNodes) {
	//	for (byte j = 0; j < nStates; j++)
	//		node.Pot.at<float>(j, 0) = expf(-node.Pot.at<float>(j, 0));
	//}

	//for (Edge &edge : m_pGraph->m_vEdges) {
	//	for (byte j = 0; j < nStates; j++)
	//		for (byte i = 0; i < nStates; i++)
	//			edge.Pot.at<float>(j, i) = expf(-edge.Pot.at<float>(j, i));
	//}




}

}