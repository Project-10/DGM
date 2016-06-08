#include "InferTRW_S.h"
#include "Graph.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	int FindMin(float *data, int len)
	{
		int res = 0;
		float minVal = data[0];
		for (int i = 1; i < len; i++)
			if (minVal < data[i]) {
				minVal = data[i];
				res = i;
			}
		return res;
	}

	void CInferTRW_S::infer(unsigned int nIt) /*const*/
	{
		const byte		nStates = m_pGraph->m_nStates;					// number of states (classes)

		// ====================================== Initialization ======================================			
		createMessages();

		size_t			nNodes = m_pGraph->getNumNodes();				// number of nodes
		vec_byte_t		res(nNodes);

		NODE*		* nodes = new NODE*[nNodes];
		float		* nPot = new float[nStates];
		float		* ePot = new float[nStates * nStates];

		// Add Nodes
		for (Node &node : m_pGraph->m_vNodes) {
			for (byte s = 0; s < nStates; s++) nPot[s] = node.Pot.at<float>(s, 0); 
			nodes[node.id] = AddNode(nPot);
		}

		// Add edges
		for (Edge &edge : m_pGraph->m_vEdges)
			if (edge.node2 > edge.node1) {
				int k = 0;
				for (byte i = 0; i < nStates; i++)
					for (byte j = 0; j < nStates; j++)
						ePot[k++] = edge.Pot.at<float>(j, i); 

				AddEdge(nodes[edge.node1], nodes[edge.node2], ePot);
			}

		// =================================== Calculating messages ==================================	

		calculateMessages(nIt);

		// =================================== Calculating beliefs ===================================	

		for (NODE *node = m_nodeFirst; node; node = node->m_next) {
			// backward edges
			for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {
				NODE *src = edge->m_tail;
				for (byte s = 0; s < nStates; s++) node->m_D[s] *= edge->m_D[src->m_solution + s * nStates];
			}

			// forward edges
			for (EDGE *edge = node->m_firstForward; edge; edge = edge->m_nextForward)
				for (byte s = 0; s < nStates; s++) node->m_D[s] *= edge->m_msg[s];

			node->m_solution = FindMin(node->m_D, nStates);
		} // All Nodes

		  // read solution
		for (Node &node : m_pGraph->m_vNodes) {
			for (byte s = 0; s < nStates; s++)
				node.Pot.at<float>(s, 0) = nodes[node.id]->m_D[s]; 
		}

		// done
		delete nodes;
		delete nPot;
		delete ePot;

		deleteMessages();
	}

	void CInferTRW_S::calculateMessages(unsigned int nIt)
	{
		const byte		  nStates = m_pGraph->m_nStates;
		float			* data = new float[nStates];
		float			* temp = new float[nStates];

		// main loop
		for (unsigned int i = 0; i < nIt; i++) {

			if (i % 5 == 0) printf("--- It: %d ---\n", i);

			// Forward pass
			for (NODE *node = m_nodeFirst; node; node = node->m_next) {					// all nodes
				memcpy(data, node->m_D, nStates * sizeof(float));						// Di = node.pot

				int nForward = 0;
				for (EDGE *edge = node->m_firstForward; edge; edge = edge->m_nextForward) {
					for (byte s = 0; s < nStates; s++) data[s] *= edge->m_msg[s];
					nForward++;
				}

				int nBackward = 0;
				for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {
					for (byte s = 0; s < nStates; s++) data[s] *= edge->m_msg[s];
					nBackward++;
				}

				for (byte s = 0; s < nStates; s++) data[s] = (float) fastPow(data[s],  1.0f / MAX(nForward, nBackward));

				// pass messages from i to nodes with higher m_ordering
				for (EDGE *edge = node->m_firstForward; edge; edge = edge->m_nextForward) {
					calculateMessage(edge, temp, data);
				}
			} // All Nodes


			  // Backward pass
			for (NODE *node = m_nodeLast; node; node = node->m_prev) {
				memcpy(data, node->m_D, nStates * sizeof(float));

				int nForward = 0;
				for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {
					for (byte s = 0; s < nStates; s++) data[s] *= edge->m_msg[s];
					nForward++;
				}

				int nBackward = 0;
				for (EDGE *edge = node->m_firstForward; edge; edge = edge->m_nextForward) {
					for (byte s = 0; s < nStates; s++) data[s] *= edge->m_msg[s];
					nBackward++;
				}

				// normalize Di
				float min = data[0];
				for (byte s = 0; s < nStates; s++) if (min < data[s]) min = data[s];
				for (byte s = 0; s < nStates; s++) data[s] /= min;
				for (byte s = 0; s < nStates; s++) data[s] = (float) fastPow(data[s], 1.0f / MAX(nForward, nBackward));

				// pass messages from i to nodes with smaller m_ordering
				for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {
					calculateMessage(edge, temp, data);
				}
			} // All Nodes
		}

		delete[] data;
		delete[] temp;
	}

	void CInferTRW_S::calculateMessage(EDGE *edge, float *temp, float *data)
	{
		const byte nStates = m_pGraph->m_nStates;

		for (byte s = 0; s < nStates; s++)
				temp[s] = data[s] / MAX(FLT_EPSILON, edge->m_msg[s]);

		for (byte y = 0; y < nStates; y++) {
			float min = temp[0] * edge->m_D[0 + y * nStates];
			for (byte x = 1; x < nStates; x++) {
				float val = temp[x] * edge->m_D[x + y * nStates];
				if (min < val)  min = val;
			}
			edge->m_msg[y] = min;
		}

		// Normalization
		float min = edge->m_msg[0];
		for (byte s = 1; s < nStates; s++) if (min < edge->m_msg[s]) min = edge->m_msg[s];
		for (byte s = 0; s < nStates; s++) edge->m_msg[s] /= min;
	}



	CInferTRW_S::NODE * CInferTRW_S::AddNode(float *data)
	{
		const byte nStates = m_pGraph->m_nStates;
		NODE *i = new NODE();
		i->m_id = m_nodeNum++;
		i->m_firstForward = NULL;
		i->m_firstBackward = NULL;
		i->m_prev = m_nodeLast;
		i->m_next = NULL;
		i->m_D = new float[nStates];
		memcpy(i->m_D, data, nStates * sizeof(float));

		if (m_nodeLast)	m_nodeLast->m_next = i;
		else m_nodeFirst = i;
		m_nodeLast = i;

		return i;
	}

	void CInferTRW_S::AddEdge(NODE *i, NODE *j, float *data)
	{
		DGM_ASSERT(i->m_id < j->m_id);

		const byte nStates = m_pGraph->m_nStates;

		EDGE *e = new EDGE();
		e->m_nextForward = i->m_firstForward;
		e->m_nextBackward = j->m_firstBackward;
		e->m_tail = i;
		e->m_head = j;

		e->m_D = new float[nStates * nStates];
		memcpy(e->m_D, data, nStates * nStates * sizeof(float));

		e->m_msg = new float[nStates];
		std::fill(e->m_msg, e->m_msg + nStates, 1.0f);				// messages

		i->m_firstForward = e;
		j->m_firstBackward = e;
	}
}

