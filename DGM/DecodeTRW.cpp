#include "DecodeTRW.h"
#include "Graph.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
vec_byte_t CDecodeTRW::decode(unsigned int nIt, Mat &lossMatrix) /*const*/
{
	byte			nStates = m_pGraph->m_nStates;					// number of states (classes)
	size_t			nNodes = m_pGraph->getNumNodes();				// number of nodes
	vec_byte_t		res(nNodes);

	DGM_IF_WARNING(!lossMatrix.empty(), "The Loss Matrix is not supported by the algorithm.");

	NODE*		* nodes = new NODE*[nNodes];
	double		* nPot = new double[nStates];
	double		* ePot = new double[nStates * nStates];

	// Add Nodes
	for (Node &node : m_pGraph->m_vNodes) {
		for (byte s = 0; s < nStates; s++) nPot[s] = -logf(MAX(FLT_EPSILON, node.Pot.at<float>(s, 0)));
		nodes[node.id] = AddNode(nPot);
	}

	// Add edges
	for (Edge &edge : m_pGraph->m_vEdges)
		if (edge.node2 > edge.node1) {
			int k = 0;
			for (byte i = 0; i < nStates; i++)
				for (byte j = 0; j < nStates; j++)
					ePot[k++] = -logf(MAX(FLT_EPSILON, edge.Pot.at<float>(j, i)));

			AddEdge(nodes[edge.node1], nodes[edge.node2], ePot);
		}


	/////////////////////// TRW-S algorithm //////////////////////
	Options	  options;
	double	  energy;
	double	  lowerBound;

	options.m_iterMax = nIt; // maximum number of iterations
	Minimize_TRW_S(options, lowerBound, energy);

	// read solution
	for (size_t n = 0; n < nNodes; n++)
		res[n] = static_cast<byte>(nodes[n]->m_solution);

	// done
	delete nodes;
	delete nPot;
	delete ePot;

	return res;
}


int FindMin(double *data, int len)
{
	int res = 0;
	double minVal = data[0];
	for (int i = 1; i < len; i++)
		if (minVal > data[i]) {
			minVal = data[i];
			res = i;
		}
	return res;
}

double ComputeAndSubtractMin(double *data, int len)
{
	double res = data[0];
	for (int i = 1; i < len; i++)
		if (res > data[i]) res = data[i];
	for (int i = 0; i < len; i++)
		data[i] -= res;
	return res;
}

CDecodeTRW::NODE * CDecodeTRW::AddNode(double *data)
{
	const byte nStates = m_pGraph->m_nStates;
	NODE *i = new NODE();
	i->m_id = m_nodeNum++;
	i->m_firstForward = NULL;
	i->m_firstBackward = NULL;
	i->m_prev = m_nodeLast;
	i->m_next = NULL;
	i->m_D = new double[nStates];
	memcpy(i->m_D, data, nStates * sizeof(double));

	if (m_nodeLast)	m_nodeLast->m_next = i;
	else m_nodeFirst = i;
	m_nodeLast = i;

	return i;
}

void CDecodeTRW::AddEdge(NODE *i, NODE *j, double *data)
{
	DGM_ASSERT(i->m_id < j->m_id);

	const byte nStates = m_pGraph->m_nStates;

	EDGE *e = new EDGE();
	e->m_nextForward = i->m_firstForward;
	e->m_nextBackward = j->m_firstBackward;
	e->m_tail = i;
	e->m_head = j;

	e->m_D = new double[nStates * nStates];
	memcpy(e->m_D, data, nStates * nStates * sizeof(double));

	e->m_msg = new double[nStates];
	memset(e->m_msg, 0, nStates * sizeof(double));

	i->m_firstForward = e;
	j->m_firstBackward = e;
}

int CDecodeTRW::Minimize_TRW_S(Options &options, double &lowerBound, double &energy)
{
	const byte nStates = m_pGraph->m_nStates;
	double	  vMin;
	double	  lowerBoundPrev;

	printf("TRW_S algorithm\n");

	double	*Di = new double[nStates];
	double	*buf = new double[nStates];

	int		iter;
	bool	lastIter = false;

	// main loop
	for (iter = 1; ; iter++) {
		if (iter >= options.m_iterMax) lastIter = true;

		////////////////////////////////////////////////
		//                forward pass                //
		////////////////////////////////////////////////
		for (NODE *node = m_nodeFirst; node; node = node->m_next) {					// all nodes
			memcpy(Di, node->m_D, nStates * sizeof(double));						// Di = node.pot
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

				UpdateMessage(edge, Di, 1.0 / MAX(nForward, nBackward), 0, buf);
			}
		} // i    All Nodes

			////////////////////////////////////////////////
			//               backward pass                //
			////////////////////////////////////////////////
		lowerBound = 0;

		for (NODE *node = m_nodeLast; node; node = node->m_prev) {
			memcpy(Di, node->m_D, nStates * sizeof(double));
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
			vMin = ComputeAndSubtractMin(Di, nStates);
			lowerBound += vMin;

			// pass messages from i to nodes with smaller m_ordering
			for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {
				assert(edge->m_head == i);

				vMin = UpdateMessage(edge, Di, 1.0 / MAX(nForward, nBackward), 1, buf);

				lowerBound += vMin;
			}
		} // i    All Nodes

			////////////////////////////////////////////////
			//          check stopping criterion          //
			////////////////////////////////////////////////

			// print lower bound and energy, if necessary
		if (lastIter || (iter >= options.m_printMinIter && (options.m_printIter < 1 || iter % options.m_printIter == 0))) {
			energy = ComputeSolutionAndEnergy();
			printf("iter %d: lower bound = %f, energy = %f\n", iter, lowerBound, energy);
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

int CDecodeTRW::Minimize_BP(Options &options, double &energy)
{
	const byte nStates = m_pGraph->m_nStates;
	NODE * i;
	NODE * j;
	EDGE * e;
	double vMin;

	printf("BP algorithm\n");

	double * Di = new double[nStates];
	double * buf = new double[nStates];

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
			memcpy(Di, i->m_D, nStates * sizeof(double));
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

				const double gamma = 1;

				UpdateMessage(e, Di, gamma, 0, buf);
			}
		}

		////////////////////////////////////////////////
		//               backward pass                //
		////////////////////////////////////////////////

		for (i = m_nodeLast; i; i = i->m_prev) {
			memcpy(Di, i->m_D, nStates * sizeof(double));
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

				const double gamma = 1;

				vMin = UpdateMessage(e, Di, gamma, 1, buf);
			}
		}

		////////////////////////////////////////////////
		//          check stopping criterion          //
		////////////////////////////////////////////////

		// print energy, if necessary
		if (lastIter ||
			(iter >= options.m_printMinIter &&
				(options.m_printIter < 1 || iter%options.m_printIter == 0))
			)
		{
			energy = ComputeSolutionAndEnergy();
			printf("iter %d: energy = %f\n", iter, energy);
		}

		// if finishFlag==true terminate
		if (lastIter) break;
	}

	delete[] Di;
	delete[] buf;

	return iter;
}

double CDecodeTRW::ComputeSolutionAndEnergy()
{
	const byte nStates = m_pGraph->m_nStates;
	double	* Di = new double[nStates];
	double	* DiBackward = new double[nStates];

	double E = 0;
	for (NODE *node = m_nodeFirst; node; node = node->m_next) {
		memcpy(DiBackward, node->m_D, nStates * sizeof(double));						// DiBackward = node.Pot
		for (EDGE *edge = node->m_firstBackward; edge; edge = edge->m_nextBackward) {	// edge: src -> node
			assert(node == edge->m_head);
			NODE *src = edge->m_tail;
			for (int k = 0; k < nStates; k++)
				DiBackward[k] += edge->m_D[src->m_solution + k * nStates];			// DiBackward = node.Pot + edge.Pot[src.sol]
		}

		// add forward edges
		memcpy(Di, DiBackward, nStates * sizeof(double));								// Di = node.Pot + edge.Pot[src.sol]

		for (EDGE *edge = node->m_firstForward; edge; edge = edge->m_nextForward)		// edge: node -> dst
			for (int k = 0; k < nStates; k++)
				Di[k] += edge->m_msg[k];												// Di = node.Pot + edge.Pot[src.sol] + edge.Msg

		node->m_solution = FindMin(Di, nStates);

		// update energy
		E += DiBackward[node->m_solution];
	} // i	All Nodes

	delete[] Di;
	delete[] DiBackward;

	return E;
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
double CDecodeTRW::UpdateMessage(EDGE *edge, double *source, double gamma, int dir, double *buf)
{
	const byte nStates = m_pGraph->m_nStates;
	double vMin;

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


