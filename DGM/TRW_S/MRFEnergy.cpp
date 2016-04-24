#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "MRFEnergy.h"

#include "instances.inc"

void DefaultErrorFn(char* msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

// Constructor
template <class T> MRFEnergy<T>::MRFEnergy(GlobalSize Kglobal, ErrorFunction errorFn)
	: m_errorFn(errorFn ? errorFn : DefaultErrorFn),
	  m_mallocBlockFirst(NULL),
	  m_nodeFirst(NULL),
	  m_nodeLast(NULL),
	  m_nodeNum(0),
	  m_edgeNum(0),
	  m_Kglobal(Kglobal),
	  m_vectorMaxSizeInBytes(0),
	  m_isEnergyConstructionCompleted(false),
	  m_buf(NULL)
{ }

// Destructor
template <class T> MRFEnergy<T>::~MRFEnergy()
{
	while (m_mallocBlockFirst) {
		MallocBlock* next = m_mallocBlockFirst->m_next;
		delete m_mallocBlockFirst;
		m_mallocBlockFirst = next;
	}
}

template <class T> typename MRFEnergy<T>::NodeId MRFEnergy<T>::AddNode(LocalSize K, NodeData data)
{
	if (m_isEnergyConstructionCompleted) m_errorFn("Error in AddNode(): graph construction completed - nodes cannot be added");
	int actualVectorSize = Vector::GetSizeInBytes(m_Kglobal, K);
	if (actualVectorSize < 0) m_errorFn("Error in AddNode() (invalid parameter?)");
	if (m_vectorMaxSizeInBytes < actualVectorSize) m_vectorMaxSizeInBytes = actualVectorSize;

	int nodeSize = sizeof(Node) - sizeof(Vector) + actualVectorSize;
	Node *i = (Node *) Malloc(nodeSize);

	i->m_K = K;
	i->m_D.Initialize(m_Kglobal, K, data);

	i->m_firstForward = NULL;
	i->m_firstBackward = NULL;
	i->m_prev = m_nodeLast;
	i->m_next = NULL;
	i->m_ordering = m_nodeNum ++;

	if (m_nodeLast)	m_nodeLast->m_next = i;
	else m_nodeFirst = i;
	m_nodeLast = i;

	return i;
}

template <class T> void MRFEnergy<T>::AddNodeData(NodeId i, NodeData data)
{
	i->m_D.Add(m_Kglobal, i->m_K, data);
}

template <class T> void MRFEnergy<T>::AddEdge(NodeId i, NodeId j, EdgeData data)
{
	if (m_isEnergyConstructionCompleted) m_errorFn("Error in AddNode(): graph construction completed - nodes cannot be added");
	int actualEdgeSize = Edge::GetSizeInBytes(m_Kglobal, i->m_K, j->m_K, data);
	if (actualEdgeSize < 0) m_errorFn("Error in AddEdge() (invalid parameter?)");
	
	int MRFedgeSize = sizeof(MRFEdge) - sizeof(Edge) + actualEdgeSize;
	MRFEdge *e = (MRFEdge*) Malloc(MRFedgeSize);

	e->m_message.Initialize(m_Kglobal, i->m_K, j->m_K, data, &i->m_D, &j->m_D);

	e->m_tail = i;
	e->m_nextForward = i->m_firstForward;
	i->m_firstForward = e;

	e->m_head = j;
	e->m_nextBackward = j->m_firstBackward;
	j->m_firstBackward = e;

	m_edgeNum ++;
}

/////////////////////////////////////////////////////////////////////////////////

template <class T> void MRFEnergy<T>::ZeroMessages()
{
	Node	* i;
	MRFEdge	* e;

	if (!m_isEnergyConstructionCompleted) CompleteGraphConstruction();
	for (i = m_nodeFirst; i; i = i->m_next)
		for (e = i->m_firstForward; e; e = e->m_nextForward)
			e->m_message.GetMessagePtr()->SetZero(m_Kglobal, i->m_K);
}

template <class T> void MRFEnergy<T>::AddRandomMessages(unsigned int random_seed, REAL min_value, REAL max_value)
{
	Node	* i;
	MRFEdge	* e;
	int 	  k;

	if (!m_isEnergyConstructionCompleted) CompleteGraphConstruction();

	srand(random_seed);

	for (i = m_nodeFirst; i; i = i->m_next) 
		for (e = i->m_firstForward; e; e = e->m_nextForward) {
			Vector* M = e->m_message.GetMessagePtr();
			for (k = 0; k<M->GetArraySize(m_Kglobal, i->m_K); k++) {
				REAL x = (REAL)( min_value + rand()/((double)RAND_MAX) * (max_value - min_value) );
				x += M->GetArrayValue(m_Kglobal, i->m_K, k);
				M->SetArrayValue(m_Kglobal, i->m_K, k, x);
			}
		}
}

/////////////////////////////////////////////////////////////////////////////////

template <class T> int MRFEnergy<T>::Minimize_TRW_S(Options &options, REAL &lowerBound, REAL &energy, REAL *min_marginals)
{
	Node	* i;
	Node	* j;
	MRFEdge	* e;
	REAL	  vMin;
	REAL	  lowerBoundPrev;

	if (!m_isEnergyConstructionCompleted) CompleteGraphConstruction();

	printf("TRW_S algorithm\n");

	SetMonotonicTrees();

	Vector	* Di = (Vector *)m_buf;
	void	* buf = (void *)(m_buf + m_vectorMaxSizeInBytes);

	int		iter = 0;
	bool	lastIter = false;

	// main loop
	for (iter = 1; ; iter++) {
		if (iter >= options.m_iterMax) lastIter = true;

		////////////////////////////////////////////////
		//                forward pass                //
		////////////////////////////////////////////////
		REAL * min_marginals_ptr = min_marginals;

		for (i = m_nodeFirst; i; i = i->m_next) {					// all nodes
			Di->Copy(m_Kglobal, i->m_K, &i->m_D);
			for (e = i->m_firstForward; e; e = e->m_nextForward)
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());

			// normalize Di, update lower bound
			// vMin = Di->ComputeAndSubtractMin(m_Kglobal, i->m_K); // do not compute lower bound
			// lowerBound += vMin;                                  // during the forward pass

			// pass messages from i to nodes with higher m_ordering
			for (e = i->m_firstForward; e; e = e->m_nextForward) {
				assert(e->m_tail == i);
				j = e->m_head;

				vMin = e->m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, e->m_gammaForward, 0, buf);

				// lowerBound += vMin; // do not compute lower bound during the forward pass
			}

			if (lastIter && min_marginals)
				min_marginals_ptr += Di->GetArraySize(m_Kglobal, i->m_K);
		} // i

		  ////////////////////////////////////////////////
		  //               backward pass                //
		  ////////////////////////////////////////////////
		lowerBound = 0;

		for (i = m_nodeLast; i; i = i->m_prev) {
			Di->Copy(m_Kglobal, i->m_K, &i->m_D);
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			for (e = i->m_firstForward; e; e = e->m_nextForward)
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());

			// normalize Di, update lower bound
			vMin = Di->ComputeAndSubtractMin(m_Kglobal, i->m_K);
			lowerBound += vMin;

			// pass messages from i to nodes with smaller m_ordering
			for (e = i->m_firstBackward; e; e = e->m_nextBackward) {
				assert(e->m_head == i);
				j = e->m_tail;

				vMin = e->m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, e->m_gammaBackward, 1, buf);

				lowerBound += vMin;
			}

			if (lastIter && min_marginals) {
				min_marginals_ptr -= Di->GetArraySize(m_Kglobal, i->m_K);
				for (int k = 0; k < Di->GetArraySize(m_Kglobal, i->m_K); k++)
					min_marginals_ptr[k] = Di->GetArrayValue(m_Kglobal, i->m_K, k);
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

template <class T> int MRFEnergy<T>::Minimize_BP(Options& options, REAL& energy, REAL* min_marginals)
{
	Node* i;
	Node* j;
	MRFEdge* e;
	REAL vMin;
	int iter;

	if (!m_isEnergyConstructionCompleted)
	{
		CompleteGraphConstruction();
	}

	printf("BP algorithm\n");

	Vector* Di = (Vector*)m_buf;
	void* buf = (void*)(m_buf + m_vectorMaxSizeInBytes);

	iter = 0;
	bool lastIter = false;

	// main loop
	for (iter = 1; ; iter++)
	{
		if (iter >= options.m_iterMax) lastIter = true;

		////////////////////////////////////////////////
		//                forward pass                //
		////////////////////////////////////////////////
		REAL* min_marginals_ptr = min_marginals;

		for (i = m_nodeFirst; i; i = i->m_next)
		{
			Di->Copy(m_Kglobal, i->m_K, &i->m_D);
			for (e = i->m_firstForward; e; e = e->m_nextForward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}

			// pass messages from i to nodes with higher m_ordering
			for (e = i->m_firstForward; e; e = e->m_nextForward)
			{
				assert(i == e->m_tail);
				j = e->m_head;

				const REAL gamma = 1;

				e->m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, gamma, 0, buf);
			}

			if (lastIter && min_marginals)
			{
				min_marginals_ptr += Di->GetArraySize(m_Kglobal, i->m_K);
			}
		}

		////////////////////////////////////////////////
		//               backward pass                //
		////////////////////////////////////////////////

		for (i = m_nodeLast; i; i = i->m_prev)
		{
			Di->Copy(m_Kglobal, i->m_K, &i->m_D);
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}
			for (e = i->m_firstForward; e; e = e->m_nextForward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}

			// pass messages from i to nodes with smaller m_ordering
			for (e = i->m_firstBackward; e; e = e->m_nextBackward)
			{
				assert(i == e->m_head);
				j = e->m_tail;

				const REAL gamma = 1;

				vMin = e->m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, gamma, 1, buf);
			}

			if (lastIter && min_marginals)
			{
				min_marginals_ptr -= Di->GetArraySize(m_Kglobal, i->m_K);
				for (int k = 0; k<Di->GetArraySize(m_Kglobal, i->m_K); k++)
				{
					min_marginals_ptr[k] = Di->GetArrayValue(m_Kglobal, i->m_K, k);
				}
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

template <class T> typename T::REAL MRFEnergy<T>::ComputeSolutionAndEnergy()
{
	Node	* i;
	Node	* j;
	MRFEdge	* e;
	REAL 	  E = 0;

	Vector	* DiBackward = (Vector*)m_buf; 					// cost of backward edges plus Di at the node
	Vector	* Di = (Vector*)(m_buf + m_vectorMaxSizeInBytes); 	// all edges plus Di at the node

	for (i = m_nodeFirst; i; i = i->m_next) {
		// Set Ebackward[ki] to be the sum of V(ki,j->m_solution) for backward edges (i,j).
		// Set Di[ki] to be the value of the energy corresponding to
		// part of the graph considered so far, assuming that nodes u
		// in this subgraph are fixed to u->m_solution

		DiBackward->Copy(m_Kglobal, i->m_K, &i->m_D);
		for (e = i->m_firstBackward; e; e = e->m_nextBackward) {
			assert(i == e->m_head);
			j = e->m_tail;
			e->m_message.AddColumn(m_Kglobal, j->m_K, i->m_K, j->m_solution, DiBackward, 0);
		}

		// add forward edges
		Di->Copy(m_Kglobal, i->m_K, DiBackward);

		for (e = i->m_firstForward; e; e = e->m_nextForward)
			Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());

		Di->ComputeMin(m_Kglobal, i->m_K, i->m_solution);

		// update energy
		E += DiBackward->GetValue(m_Kglobal, i->m_K, i->m_solution);
	}

	return E;
}

/////////////////////////////////////////////////////////////////////////////////

template <class T> void MRFEnergy<T>::CompleteGraphConstruction()
{
	Node	* i;
	Node	* j;
	MRFEdge	* e;
	MRFEdge	* ePrev;

	if (m_isEnergyConstructionCompleted) m_errorFn("Fatal error in CompleteGraphConstruction");

	printf("Completing graph construction... ");

	if (m_buf) m_errorFn("CompleteGraphConstruction(): fatal error");

	m_buf = (char *) Malloc(m_vectorMaxSizeInBytes + 
		( m_vectorMaxSizeInBytes > Edge::GetBufSizeInBytes(m_vectorMaxSizeInBytes) ?
		  m_vectorMaxSizeInBytes : Edge::GetBufSizeInBytes(m_vectorMaxSizeInBytes) ) );

	// set forward and backward edges properly
#ifdef _DEBUG
	int ordering;
	for (i = m_nodeFirst, ordering = 0; i; i = i->m_next, ordering++) {
		if ( (i->m_ordering != ordering)
		  || (i->m_ordering == 0 && i->m_prev)
		  || (i->m_ordering != 0 && i->m_prev->m_ordering != ordering-1) )
		{
			m_errorFn("CompleteGraphConstruction(): fatal error (wrong ordering)");
		}
	}
	if (ordering != m_nodeNum) m_errorFn("CompleteGraphConstruction(): fatal error");

#endif
	for (i = m_nodeFirst; i; i = i->m_next) i->m_firstBackward = NULL;
	for (i = m_nodeFirst; i; i = i->m_next) {
		ePrev = NULL;
		for (e = i->m_firstForward; e; ) {
			assert(i == e->m_tail);
			j = e->m_head;

			if (i->m_ordering < j->m_ordering) {
				e->m_nextBackward = j->m_firstBackward;
				j->m_firstBackward = e;

				ePrev = e;
				e = e->m_nextForward;
			} else {
				e->m_message.Swap(m_Kglobal, i->m_K, j->m_K);
				e->m_tail = j;
				e->m_head = i;

				MRFEdge* eNext = e->m_nextForward;

				if (ePrev) ePrev->m_nextForward = e->m_nextForward;
				else i->m_firstForward = e->m_nextForward;

				e->m_nextForward = j->m_firstForward;
				j->m_firstForward = e;

				e->m_nextBackward = i->m_firstBackward;
				i->m_firstBackward = e;

				e = eNext;
			}
		}
	}

	m_isEnergyConstructionCompleted = true;

	// ZeroMessages();

	printf("done\n");
}

template <class T> void MRFEnergy<T>::SetMonotonicTrees()
{
	Node	* i;
	MRFEdge	* e;

	if (!m_isEnergyConstructionCompleted) CompleteGraphConstruction();

	for (i = m_nodeFirst; i; i = i->m_next) {
		int nForward = 0, nBackward = 0;
		for (e = i->m_firstForward; e; e = e->m_nextForward) 	nForward++;
		for (e = i->m_firstBackward; e; e = e->m_nextBackward)	nBackward++;

		int ni = (nForward > nBackward) ? nForward : nBackward;

		REAL mu = (REAL)1 / ni;
		for (e = i->m_firstBackward; e; e = e->m_nextBackward)	e->m_gammaBackward = mu;
		for (e = i->m_firstForward; e; e = e->m_nextForward)	e->m_gammaForward = mu;
	}
}