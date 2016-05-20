#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "MRFEnergy.h"

#include "..\..\include\types.h"
#include "..\..\include\macroses.h"

double ComputeMin(double *data, int len, int &kMin)
{
	double res = data[0];
	kMin = 0;
	for (int i = 1; i < len; i++)
		if (res > data[i]) {
			res = data[i];
			kMin = i;
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

// Constructor
MRFEnergy::MRFEnergy(int nStates)
	: m_nStates(nStates),
	  m_nodeFirst(NULL),
	  m_nodeLast(NULL),
	  m_nodeNum(0),
	  m_isEnergyConstructionCompleted(false)
{
	m_buf = new double[2 * m_nStates];
}

// Destructor
MRFEnergy::~MRFEnergy()
{
	// TODO: free memory
	delete[] m_buf;
}

MRFEnergy::Node * MRFEnergy::AddNode(double *data)
{
	DGM_ASSERT_MSG(!m_isEnergyConstructionCompleted, "Error in AddNode(): graph construction completed - nodes cannot be added");
	
	Node *i = new Node();
	i->m_id = m_nodeNum++;
	i->m_firstForward = NULL;
	i->m_firstBackward = NULL;
	i->m_prev = m_nodeLast;
	i->m_next = NULL;
	i->m_D = new double[m_nStates];						
	memcpy(i->m_D, data, m_nStates * sizeof(double));

	if (m_nodeLast)	m_nodeLast->m_next = i;
	else m_nodeFirst = i;
	m_nodeLast = i;

	return i;
}

void MRFEnergy::AddEdge(Node *i, Node *j, double *data)
{
	DGM_ASSERT_MSG (!m_isEnergyConstructionCompleted, "Error in AddNode(): graph construction completed - nodes cannot be added");
	DGM_ASSERT(i->m_id < j->m_id);

	MRFEdge *e = new MRFEdge();
	e->m_nextForward = i->m_firstForward;
	e->m_nextBackward = j->m_firstBackward;
	e->m_tail = i;
	e->m_head = j;
	e->m_message.Initialize(m_nStates, data); // m_dir = 0; m_data = data; m_message = 0;
	
	i->m_firstForward = e;
	j->m_firstBackward = e;
}

int MRFEnergy::Minimize_TRW_S(Options &options, double &lowerBound, double &energy, double *min_marginals)
{
	Node	* i;
	Node	* j;
	MRFEdge	* e;
	double	  vMin;
	double	  lowerBoundPrev;

	if (!m_isEnergyConstructionCompleted) CompleteGraphConstruction();

	printf("TRW_S algorithm\n");

	SetMonotonicTrees();

	double	* Di =  m_buf;
	double	* buf = m_buf + m_nStates;

	int		iter	 = 0;
	bool	lastIter = false;

	// main loop
	for (iter = 1; ; iter++) {
		if (iter >= options.m_iterMax) lastIter = true;

		////////////////////////////////////////////////
		//                forward pass                //
		////////////////////////////////////////////////
		double * min_marginals_ptr = min_marginals;

		for (i = m_nodeFirst; i; i = i->m_next) {					// all nodes
			memcpy(Di, i->m_D, m_nStates * sizeof(double));
			for (e = i->m_firstForward; e; e = e->m_nextForward)
				for (int k = 0; k < m_nStates; k++)
					Di[k] += e->m_message.m_message[k];
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
				for (int k = 0; k < m_nStates; k++)
					Di[k] += e->m_message.m_message[k];

			// normalize Di, update lower bound
			// vMin = Di->ComputeAndSubtractMin(m_Kglobal, i->m_K); // do not compute lower bound
			// lowerBound += vMin;                                  // during the forward pass

			// pass messages from i to nodes with higher m_ordering
			for (e = i->m_firstForward; e; e = e->m_nextForward) {
				assert(e->m_tail == i);
				j = e->m_head;

				vMin = e->m_message.UpdateMessage(m_nStates, Di, e->m_gammaForward, 0, buf);

				// lowerBound += vMin; // do not compute lower bound during the forward pass
			}

			if (lastIter && min_marginals)
				min_marginals_ptr += m_nStates;
		} // i

		  ////////////////////////////////////////////////
		  //               backward pass                //
		  ////////////////////////////////////////////////
		lowerBound = 0;

		for (i = m_nodeLast; i; i = i->m_prev) {
			memcpy(Di, i->m_D, m_nStates * sizeof(double));
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
				for (int k = 0; k < m_nStates; k++)
					Di[k] += e->m_message.m_message[k];
			for (e = i->m_firstForward; e; e = e->m_nextForward)
				for (int k = 0; k < m_nStates; k++)
					Di[k] += e->m_message.m_message[k];

			// normalize Di, update lower bound
			vMin = ComputeAndSubtractMin(Di, m_nStates);
			lowerBound += vMin;

			// pass messages from i to nodes with smaller m_ordering
			for (e = i->m_firstBackward; e; e = e->m_nextBackward) {
				assert(e->m_head == i);
				j = e->m_tail;

				vMin = e->m_message.UpdateMessage(m_nStates, Di, e->m_gammaBackward, 1, buf);

				lowerBound += vMin;
			}

			if (lastIter && min_marginals) {
				min_marginals_ptr -= m_nStates;
				for (int k = 0; k < m_nStates; k++)
					min_marginals_ptr[k] = Di[k];
			}
		}

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

	return iter;
}

int MRFEnergy::Minimize_BP(Options &options, double &energy, double *min_marginals)
{
	Node* i;
	Node* j;
	MRFEdge* e;
	double vMin;
	int iter;

	if (!m_isEnergyConstructionCompleted)
		CompleteGraphConstruction();

	printf("BP algorithm\n");

	double * Di =  m_buf;
	double * buf = m_buf + m_nStates;

	iter = 0;
	bool lastIter = false;

	// main loop
	for (iter = 1; ; iter++)
	{
		if (iter >= options.m_iterMax) lastIter = true;

		////////////////////////////////////////////////
		//                forward pass                //
		////////////////////////////////////////////////
		double* min_marginals_ptr = min_marginals;

		for (i = m_nodeFirst; i; i = i->m_next)
		{
			memcpy(Di, i->m_D, m_nStates * sizeof(double));
			for (e = i->m_firstForward; e; e = e->m_nextForward)
				for (int k = 0; k < m_nStates; k++)
					Di[k] += e->m_message.m_message[k];
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
				for (int k = 0; k < m_nStates; k++)
					Di[k] += e->m_message.m_message[k];


			// pass messages from i to nodes with higher m_ordering
			for (e = i->m_firstForward; e; e = e->m_nextForward)
			{
				assert(i == e->m_tail);
				j = e->m_head;

				const double gamma = 1;

				e->m_message.UpdateMessage(m_nStates, Di, gamma, 0, buf);
			}

			if (lastIter && min_marginals)
				min_marginals_ptr += m_nStates;
		}

		////////////////////////////////////////////////
		//               backward pass                //
		////////////////////////////////////////////////

		for (i = m_nodeLast; i; i = i->m_prev) {
			memcpy(Di, i->m_D, m_nStates * sizeof(double));
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
				for (int k = 0; k < m_nStates; k++)
					Di[k] += e->m_message.m_message[k];
			for (e = i->m_firstForward; e; e = e->m_nextForward)
				for (int k = 0; k < m_nStates; k++)
					Di[k] += e->m_message.m_message[k];

			// pass messages from i to nodes with smaller m_ordering
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
			{
				assert(i == e->m_head);
				j = e->m_tail;

				const double gamma = 1;

				vMin = e->m_message.UpdateMessage(m_nStates, Di, gamma, 1, buf);
			}

			if (lastIter && min_marginals) {
				min_marginals_ptr -= m_nStates;
				for (int k = 0; k < m_nStates; k++)
					min_marginals_ptr[k] = Di[k];
			}
		}

		////////////////////////////////////////////////
		//          check stopping criterion          //
		////////////////////////////////////////////////

		// print energy, if necessary
		if (lastIter ||
			(iter >= options.m_printMinIter &&
				(options.m_printIter<1 || iter%options.m_printIter == 0))
			)
		{
			energy = ComputeSolutionAndEnergy();
			printf("iter %d: energy = %f\n", iter, energy);
		}

		// if finishFlag==true terminate
		if (lastIter) break;
	}

	return iter;
}

double MRFEnergy::ComputeSolutionAndEnergy()
{
	Node	* i;
	Node	* j;
	MRFEdge	* e;
	double 	  E = 0;

	double	* DiBackward = m_buf; 					// cost of backward edges plus Di at the node
	double	* Di = m_buf + m_nStates; 				// all edges plus Di at the node

	for (i = m_nodeFirst; i; i = i->m_next) {
		// Set Ebackward[ki] to be the sum of V(ki,j->m_solution) for backward edges (i,j).
		// Set Di[ki] to be the value of the energy corresponding to
		// part of the graph considered so far, assuming that nodes u
		// in this subgraph are fixed to u->m_solution

		memcpy(DiBackward, i->m_D, m_nStates * sizeof(double));
		for (e = i->m_firstBackward; e; e = e->m_nextBackward) {
			assert(i == e->m_head);
			j = e->m_tail;
			e->m_message.AddColumn(m_nStates, j->m_solution, DiBackward, 0);
		}

		// add forward edges
		memcpy(Di, DiBackward, m_nStates * sizeof(double));

		for (e = i->m_firstForward; e; e = e->m_nextForward)
			for (int k = 0; k < m_nStates; k++)
				Di[k] += e->m_message.m_message[k];

		ComputeMin(Di, m_nStates, i->m_solution);

		// update energy
		E += DiBackward[i->m_solution];
	}

	return E;
}

/////////////////////////////////////////////////////////////////////////////////

void MRFEnergy::CompleteGraphConstruction()
{
	DGM_ASSERT_MSG (!m_isEnergyConstructionCompleted, "Fatal error in CompleteGraphConstruction");
	printf("Completing graph construction... ");

	// set forward and backward edges properly
	for (Node *i = m_nodeFirst; i; i = i->m_next) i->m_firstBackward = NULL;

	for (Node *i = m_nodeFirst; i; i = i->m_next) {
		MRFEdge *ePrev = NULL;
		for (MRFEdge *e = i->m_firstForward; e; ) {		// e : i -> j
			DGM_ASSERT(i == e->m_tail);
			Node *j = e->m_head;
			DGM_ASSERT(i->m_id < j->m_id);				// ordering

			e->m_nextBackward = j->m_firstBackward;
			j->m_firstBackward = e;

			ePrev = e;
			e = e->m_nextForward;
		}
	}

	m_isEnergyConstructionCompleted = true;

	printf("done\n");
}

void MRFEnergy::SetMonotonicTrees()
{
	Node	* i;
	MRFEdge	* e;

	if (!m_isEnergyConstructionCompleted) CompleteGraphConstruction();

	for (i = m_nodeFirst; i; i = i->m_next) {
		int nForward = 0, nBackward = 0;
		for (e = i->m_firstForward; e; e = e->m_nextForward) 	nForward++;
		for (e = i->m_firstBackward; e; e = e->m_nextBackward)	nBackward++;

		int ni = (nForward > nBackward) ? nForward : nBackward;

		double mu = (double) 1 / ni;
		for (e = i->m_firstBackward; e; e = e->m_nextBackward)	e->m_gammaBackward = mu;
		for (e = i->m_firstForward; e; e = e->m_nextForward)	e->m_gammaForward = mu;
	}
}