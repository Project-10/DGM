#include "DecodeTRW.h"
#include "Graph.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
byte FindMin1(float *data, byte len)
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

byte FindMax1(float *data, byte len)
{
	byte res = 0;
	float maxVal = data[0];
	for (byte i = 1; i < len; i++)
		if (maxVal < data[i]) {
			maxVal = data[i];
			res = i;
		}
	return res;
}

float ComputeAndSubtractMin1(float *data, int len)
{
	float res = data[0];
	for (int i = 1; i < len; i++)
		if (res > data[i]) res = data[i];
	for (int i = 0; i < len; i++)
		data[i] -= res;
	return res;
}

vec_byte_t CDecodeTRW::decode(unsigned int nIt, Mat &lossMatrix) /*const*/
{
	byte			nStates = m_pGraph->m_nStates;					// number of states (classes)
	size_t			nNodes = m_pGraph->getNumNodes();				// number of nodes
	vec_byte_t		res(nNodes);

	DGM_IF_WARNING(!lossMatrix.empty(), "The Loss Matrix is not supported by the algorithm.");

	NODE*		* nodes = new NODE*[nNodes];

	TransformPotentials();

	// Add Nodes
	for (Node &node : m_pGraph->m_vNodes) 
		nodes[node.id] = AddNode((float *) node.Pot.data);

	// Add edges
	for (Edge &edge : m_pGraph->m_vEdges)
		if (edge.node2 > edge.node1) 
			AddEdge(nodes[edge.node1], nodes[edge.node2], (float *) edge.Pot.data);


	/////////////////////// TRW-S algorithm //////////////////////
	Options	  options;
	float	  lowerBound;

	options.m_iterMax = nIt; // maximum number of iterations
	Minimize_TRW_S(options, lowerBound);

	ComputeSolution();

	// read solution
	//for (size_t n = 0; n < nNodes; n++)
	//	res[n] = static_cast<byte>(nodes[n]->m_solution);

	for (NODE *node = m_nodeFirst; node; node = node->m_next) {					// all nodes
		for (byte s = 0; s < nStates; s++)
			m_pGraph->m_vNodes[node->m_id].Pot.at<float>(s, 0) = node->m_D[s];
	}
	TransformPotentialsBack();


	Mat pot;
	for (size_t n = 0; n < nNodes; n++) {
		m_pGraph->getNode(n, pot);
		Point extremumLoc;
		minMaxLoc(pot, NULL, NULL, NULL, &extremumLoc);
		res[n] = static_cast<byte>(extremumLoc.y);
	}
		
	
	// done
	delete nodes;

	return res;
}

// ================================ PRIVATE ================================

void CDecodeTRW::TransformPotentials()
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

void CDecodeTRW::TransformPotentialsBack()
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

CDecodeTRW::NODE * CDecodeTRW::AddNode(float *data)
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

void CDecodeTRW::AddEdge(NODE *i, NODE *j, float *data)
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
	memset(e->m_msg, 0, nStates * sizeof(float));

	i->m_firstForward = e;
	j->m_firstBackward = e;
}

int CDecodeTRW::Minimize_TRW_S(Options &options, float &lowerBound)
{
	const byte nStates = m_pGraph->m_nStates;
	float	  vMin;
	float	  lowerBoundPrev;

	printf("TRW_S algorithm\n");

	float	*Di = new float[nStates];
	float	*buf = new float[nStates];

	int		iter;
	bool	lastIter = false;

	// main loop
	for (iter = 1; ; iter++) {
		if (iter >= options.m_iterMax) lastIter = true;

		////////////////////////////////////////////////
		//                forward pass                //
		////////////////////////////////////////////////
		for (NODE *node = m_nodeFirst; node; node = node->m_next) {					// all nodes
			memcpy(Di, node->m_D, nStates * sizeof(float));						// Di = node.pot
			int nForward = 0;
			for (EDGE *edge = node->m_firstForward; edge; edge = edge->m_nextForward) {
				for (int k = 0; k < nStates; k++)
					Di[k] += edge->m_msg[k];
				nForward++;
			}
			int nBackward = 0;
			for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {
				for (int k = 0; k < nStates; k++)
					Di[k] += edge->m_msg[k];
				nBackward++;
			}
			// pass messages from i to nodes with higher m_ordering
			for (EDGE *edge = node->m_firstForward; edge; edge = edge->m_nextForward) {
				assert(edge->m_tail == i);

				UpdateMessage(edge, Di, 1.0f / MAX(nForward, nBackward), 0, buf);
			}
		} // i    All Nodes

			////////////////////////////////////////////////
			//               backward pass                //
			////////////////////////////////////////////////
		lowerBound = 0;

		for (NODE *node = m_nodeLast; node; node = node->m_prev) {
			memcpy(Di, node->m_D, nStates * sizeof(float));
			int nForward = 0;
			for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {
				for (int k = 0; k < nStates; k++)
					Di[k] += edge->m_msg[k];
				nForward++;
			}
			int nBackward = 0;
			for (EDGE *edge = node->m_firstForward; edge; edge = edge->m_nextForward) {
				for (int k = 0; k < nStates; k++)
					Di[k] += edge->m_msg[k];
				nBackward++;
			}

			// normalize Di, update lower bound
			vMin = ComputeAndSubtractMin1(Di, nStates);
			lowerBound += vMin;

			// pass messages from i to nodes with smaller m_ordering
			for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {
				assert(edge->m_head == i);

				vMin = UpdateMessage(edge, Di, 1.0f / MAX(nForward, nBackward), 1, buf);

				lowerBound += vMin;
			}
		} // i    All Nodes

			////////////////////////////////////////////////
			//          check stopping criterion          //
			////////////////////////////////////////////////

			// print lower bound, if necessary
		if (lastIter || (iter >= options.m_printMinIter && (options.m_printIter < 1 || iter % options.m_printIter == 0))) {
			//ComputeSolution();
			printf("iter %d: lower bound = %f\n", iter, lowerBound);
		}

		if (lastIter) break;

		// check convergence of lower bound
		if (options.m_eps >= 0) {
			if (iter > 1 && lowerBound - lowerBoundPrev <= options.m_eps)
				lastIter = true;
			lowerBoundPrev = lowerBound;
		}
	}

	delete[] Di;
	delete[] buf;

	return iter;
}

int CDecodeTRW::Minimize_BP(Options &options)
{
	const byte nStates = m_pGraph->m_nStates;
	NODE * i;
	NODE * j;
	EDGE * e;
	float vMin;

	printf("BP algorithm\n");

	float * Di = new float[nStates];
	float * buf = new float[nStates];

	int	 iter;
	bool lastIter = false;

	// main loop
	for (iter = 1; ; iter++)
	{
		if (iter >= options.m_iterMax) lastIter = true;

		////////////////////////////////////////////////
		//                forward pass                //
		////////////////////////////////////////////////
		for (i = m_nodeFirst; i; i = i->m_next)
		{
			memcpy(Di, i->m_D, nStates * sizeof(float));
			for (e = i->m_firstForward; e; e = e->m_nextForward)
				for (int k = 0; k < nStates; k++)
					Di[k] += e->m_msg[k];
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
				for (int k = 0; k < nStates; k++)
					Di[k] += e->m_msg[k];


			// pass messages from i to nodes with higher m_ordering
			for (e = i->m_firstForward; e; e = e->m_nextForward)
			{
				assert(i == e->m_tail);
				j = e->m_head;

				const float gamma = 1;

				UpdateMessage(e, Di, gamma, 0, buf);
			}
		}

		////////////////////////////////////////////////
		//               backward pass                //
		////////////////////////////////////////////////

		for (i = m_nodeLast; i; i = i->m_prev) {
			memcpy(Di, i->m_D, nStates * sizeof(float));
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
				for (int k = 0; k < nStates; k++)
					Di[k] += e->m_msg[k];
			for (e = i->m_firstForward; e; e = e->m_nextForward)
				for (int k = 0; k < nStates; k++)
					Di[k] += e->m_msg[k];

			// pass messages from i to nodes with smaller m_ordering
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
			{
				assert(i == e->m_head);
				j = e->m_tail;

				const float gamma = 1;

				vMin = UpdateMessage(e, Di, gamma, 1, buf);
			}
		}

		////////////////////////////////////////////////
		//          check stopping criterion          //
		////////////////////////////////////////////////

		// print energy, if necessary
		if (lastIter ||	(iter >= options.m_printMinIter && (options.m_printIter < 1 || iter%options.m_printIter == 0))) {
			ComputeSolution();
			printf("iter %d\n", iter);
		}

		// if finishFlag==true terminate
		if (lastIter) break;
	}

	delete[] Di;
	delete[] buf;

	return iter;
}

void CDecodeTRW::ComputeSolution()
{
	const byte nStates = m_pGraph->m_nStates;

	for (NODE *node = m_nodeFirst; node; node = node->m_next) {
		for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {	// edge: src -> node
			assert(node == edge->m_head);
			NODE *src = edge->m_tail;
			for (int k = 0; k < nStates; k++)
				node->m_D[k] += edge->m_D[src->m_solution + k * nStates];				// Di = node.Pot + edge.Pot[src.sol]
		}

		// add forward edges
		for (EDGE *edge = node->m_firstForward; edge; edge = edge->m_nextForward)		// edge: node -> dst
			for (int k = 0; k < nStates; k++)
				node->m_D[k] += edge->m_msg[k];											// Di = node.Pot + edge.Pot[src.sol] + edge.Msg

		node->m_solution = FindMin1(node->m_D, nStates);

	} // i	All Nodes
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
float CDecodeTRW::UpdateMessage(EDGE *edge, float *source, float gamma, int dir, float *buf)
{
	const byte nStates = m_pGraph->m_nStates;
	float vMin;

	for (int ksource = 0; ksource < nStates; ksource++)
		buf[ksource] = gamma * source[ksource] - edge->m_msg[ksource];

	if (dir == 0) {
		for (int kdest = 0; kdest < nStates; kdest++) {
			vMin = buf[0] + edge->m_D[0 + kdest * nStates];
			for (int ksource = 1; ksource < nStates; ksource++) {
				if (vMin > buf[ksource] + edge->m_D[ksource + kdest * nStates])
					vMin = buf[ksource] + edge->m_D[ksource + kdest * nStates];
			}
			edge->m_msg[kdest] = vMin;
		}
	}
	else {
		for (int kdest = 0; kdest < nStates; kdest++) {
			vMin = buf[0] + edge->m_D[kdest + 0 * nStates];
			for (int ksource = 1; ksource < nStates; ksource++) {
				if (vMin > buf[ksource] + edge->m_D[kdest + ksource * nStates])
					vMin = buf[ksource] + edge->m_D[kdest + ksource * nStates];
			}
			edge->m_msg[kdest] = vMin;
		}
	}

	vMin = edge->m_msg[0];
	for (int kdest = 1; kdest < nStates; kdest++)
		if (vMin > edge->m_msg[kdest])
			vMin = edge->m_msg[kdest];

	for (int kdest = 0; kdest < nStates; kdest++)
		edge->m_msg[kdest] -= vMin;


	return vMin;



}
}


