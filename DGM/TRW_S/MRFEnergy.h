/******************************************************************
Vladimir Kolmogorov, 2005
vnk@microsoft.com

(c) Microsoft Corporation. All rights reserved. 
*******************************************************************/
#pragma once

#include "typeGeneral.h"

// After MRFEnergy is allocated, there are two phases:
// 1. Energy construction. Only AddNode() and AddEdge() may be called.
// Any call ZeroMessages(), Minimize_TRW_S() or Minimize_BP() completes graph construction; 
// MRFEnergy goes to the second phase:
// 2. Only functions ZeroMessages(), Minimize_TRW_S(), Minimize_BP()
// or GetSolution() may be called. (The last function can be called only after Minimize_TRW_S() or Minimize_BP()).

template <class T> class MRFEnergy
{
public:
	struct Node;

	// Constructor. Function errorFn is called with an error message, if an error occurs.
	MRFEnergy(void);
	~MRFEnergy(void);

	//////////////////////////////////////////////////////////
	//                 Energy construction                  //
	//////////////////////////////////////////////////////////

	// Adds a node with parameters K and data 
	// (see the corresponding message*.h file for description).
	// Note: information in data is copied into internal memory.
	// Cannot be called after energy construction is completed.
	Node * AddNode(int K, double * data);

	// Adds an edge between i and j. data determins edge parameters
	// (see the corresponding message*.h file for description).
	// Note: information in data is copied into internal memory.
	// Cannot be called after energy construction is completed.
	void AddEdge(Node * i, Node * j, double * data);

	//////////////////////////////////////////////////////////
	//                Energy construction end               //
	//////////////////////////////////////////////////////////

	// Clears all messages. Completes energy construction (if not completed yet).
	void ZeroMessages(void);

	// Adds to all message entries a value drawn uniformly from [min_value, max_value].
	// Normally, min_value can be set to 0 (except for TypeBinaryFast, in which case min_value = -max_value)
	void AddRandomMessages(unsigned int random_seed, double min_value, double max_value);

	// The structure below specifies (1) stopping criteria and 
	// (2) how often to compute solution and print its energy.
	struct Options
	{
		Options()
		{
			// default parameters
			m_eps			= -1;		// not used
			m_iterMax		= 1000000;
			m_printIter		= 5;		// After 10 iterations start printing the lower bound
			m_printMinIter	= 10;		// and the energy every 5 iterations.
		}

		// stopping criterion
		double		m_eps;				// stop if the increase in the lower bound during one iteration is less or equal than m_eps.
										// Used only if m_eps >= 0, and only for TRW-S algorithm.
		int			m_iterMax;			// maximum number of iterations

		// Option for printing lower bound and the energy.
		// Note: computing solution and its energy is slow
		// (it is comparable to the cost of one iteration).
		int		m_printIter;			// print lower bound and energy every m_printIter iterations
		int		m_printMinIter;			// do not print lower bound and energy before m_printMinIter iterations
	};

	// Returns number of iterations. Sets lowerBound and energy.
	// If the user provides array min_marginals, then the code
	// sets this array accordingly. (The size of the array depends on the type
	// used. Normally, it's (# nodes)*(# labels). Exception: for TypeBinaryFast it's (# nodes).
	int Minimize_TRW_S(Options& options, double& lowerBound, double& energy, double* min_marginals = NULL);

	// Returns number of iterations. Sets energy.
	int Minimize_BP(Options& options, double& energy, double* min_marginals = NULL);

	// Returns an integer in [0,Ki). Can be called only after Minimize().
	int GetSolution(Node * i) { return i->m_solution; }

	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////
	//                   Implementation                     //
	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////
private:

	typedef typename T::Vector Vector;
	typedef typename T::Edge   Edge;

	struct MRFEdge;
	struct MallocBlock;

	MallocBlock		* m_mallocBlockFirst;
	Node			* m_nodeFirst;
	Node			* m_nodeLast;
	int				  m_nodeNum;
	int				  m_edgeNum;
	int				  m_vectorMaxSizeInBytes;
	bool			  m_isEnergyConstructionCompleted;
	char			* m_buf; 				// buffer of size m_vectorMaxSizeInBytes + max(m_vectorMaxSizeInBytes, Edge::GetBufSizeInBytes(m_vectorMaxSizeInBytes))

	void CompleteGraphConstruction(void); 	// nodes and edges cannot be added after calling this function
	void SetMonotonicTrees(void);

	double ComputeSolutionAndEnergy(void); 	// sets Node::m_solution, returns value of the energy

	struct Node
	{
		int			  m_ordering; 		///< unique integer in [0,m_nodeNum-1)
		MRFEdge		* m_firstForward; 	///< first edge going to nodes with greater m_ordering
		MRFEdge		* m_firstBackward; 	///< first edge going to nodes with smaller m_ordering
		Node		* m_prev; 			///< previous and next
		Node		* m_next; 			///< nodes according to m_ordering
		int			  m_solution; 		///< integer in [0,m_D.m_K)
		int			  m_K; 				///< local information about number of labels
		Vector		  m_D;				///< must be the last member in the struct since its size is not fixed
	};

	struct MRFEdge
	{
		MRFEdge		* m_nextForward; 	///< next forward edge with the same tail
		MRFEdge		* m_nextBackward; 	///< next backward edge with the same head
		Node		* m_tail;
		Node		* m_head;
		double		  m_gammaForward; 	///< = rho_{ij} / rho_{i} where i=m_tail, j=m_head
		double		  m_gammaBackward; 	///< = rho_{ij} / rho_{j} where i=m_tail, j=m_head
		Edge		  m_message; 		///< must be the last member in the struct since its size is not fixed.
									  	///< Stores edge information and either forward or backward message.
									  	///< Most of the time it's the backward message; it gets replaced
									  	///< by the forward message only temporarily inside Minimize_TRW_S() and Minimize_BP().
	};

	// Use our own Malloc since 
	// (a) new in C++ is slow and allocates minimum memory of 64 bytes (in Visual C++)
	// (b) we want simple (one function) deallocation instead of going through every allocated element
	struct MallocBlock
	{
		static const int minBlockSizeInBytes = 4096 - 3*sizeof(void*);
		MallocBlock	* m_next;
		char		* m_current;		// first element of available memory in this block
		char		* m_last;			// first element outside of allocated memory for this block
	};
	
	inline char * Malloc(int bytesNum) 
	{
		if (!m_mallocBlockFirst || m_mallocBlockFirst->m_current + bytesNum > m_mallocBlockFirst->m_last) {
			int size = (bytesNum > MallocBlock::minBlockSizeInBytes) ? bytesNum : MallocBlock::minBlockSizeInBytes;
			MallocBlock* b = (MallocBlock*) new char[sizeof(MallocBlock) + size];
			if (!b) {
				printf("Not enough memory");
				exit(-1);
			}
			b->m_current = (char*)b + sizeof(MallocBlock);
			b->m_last = b->m_current + size;

			b->m_next = m_mallocBlockFirst;
			m_mallocBlockFirst = b;
		}

		char* ptr = m_mallocBlockFirst->m_current;
		m_mallocBlockFirst->m_current += bytesNum;
		return ptr;
	}
};
