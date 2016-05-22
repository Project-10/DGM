// Microsoft Research TRW class interface
// Written by Sergey G. Kosov in 2013 for Project X
#pragma once

#include "decode.h"

namespace DirectGraphicalModels
{
	// ==================== Microsoft TRW Decode Class ==================
	/**
	* @ingroup moduleDecode
	* @brief Microsoft Tree-reweighted decoding class
	* @details This class is based on the <a href="http://research.microsoft.com/en-us/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/" target="_blank">Tree-reweighted message passing algorithm for energy minimization</a> v.1.3 
	* (a modification of a max-poduct LBP algorithm), described in the paper <a href="http://pub.ist.ac.at/~vnk/papers/TRW-S-PAMI.pdf" target="_blank">Convergent Tree-reweighted Message Passing for Energy Minimization</a>
	* @note This class supports only undirected arcs with symmetric potentials in the graph.
	* @warning Do not use this class with directed or mixed graphs
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CDecodeTRW : public CDecode
	{
	public:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/	
		DllExport CDecodeTRW(CGraph *pGraph) : CDecode(pGraph), m_nodeFirst(NULL), m_nodeLast(NULL), m_nodeNum(0) {};
		DllExport virtual ~CDecodeTRW(void) {};

		/**
		* @brief Aproximate decoding
		* @param nIt Number of iterations
		* @param lossMatrix is not used
		* @return The most probable configuration
		*/
		DllExport virtual vec_byte_t decode(unsigned int nIt = 10, Mat &lossMatrix = Mat())/* const*/;

	
	private:
		/******************************************************************
		Vladimir Kolmogorov, 2005
		(c) Microsoft Corporation. All rights reserved.
		*******************************************************************/
		struct NODE;
		struct EDGE;
		struct Options;
		
		void	  TransformPotentials(void);													// Changes the potentials; pot = -log(pot)
		void	  TransformPotentialsBack(void);												// Changes the potentials; pot = exp(-pot)

		NODE	* AddNode(float *data);
		void	  AddEdge(NODE *i, NODE *j, float *data);
		
		int		  Minimize_TRW_S(Options& options, float& lowerBound);							// Returns number of iterations. Sets lowerBound and energy.
		int		  Minimize_BP(Options& options);												// Returns number of iterations. Sets energy.
		
		void	  ComputeSolution(void); 														// sets Node::m_solution
		float	  UpdateMessage(EDGE *edge, float *source, float gamma, int dir, float *buf);

		NODE	* m_nodeFirst;
		NODE	* m_nodeLast;
		int		  m_nodeNum;

		struct Options {
			Options()
			{
				// default parameters
				m_eps = -1;				// not used
				m_iterMax = 1000000;
				m_printIter = 5;		// After 10 iterations start printing the lower bound
				m_printMinIter = 10;	// and the energy every 5 iterations.
			}

			// stopping criterion
			float	m_eps;				// stop if the increase in the lower bound during one iteration is less or equal than m_eps. Used only if m_eps >= 0, and only for TRW-S algorithm.
			int		m_iterMax;			// maximum number of iterations
			int		m_printIter;		// print lower bound and energy every m_printIter iterations
			int		m_printMinIter;		// do not print lower bound and energy before m_printMinIter iterations
		};

		struct NODE {
			int			  m_id; 			///< unique integer in [0,m_nodeNum-1)
			EDGE		* m_firstForward; 	///< first edge going to nodes with greater m_ordering
			EDGE		* m_firstBackward; 	///< first edge going to nodes with smaller m_ordering
			NODE		* m_prev; 			///< previous and next
			NODE		* m_next; 			///< nodes according to m_ordering
			int			  m_solution; 		///< integer in [0,m_D.m_K)
			float		* m_D;				///< node potential
		};

		struct EDGE {
			EDGE		* m_nextForward; 	///< next forward edge with the same tail
			EDGE		* m_nextBackward; 	///< next backward edge with the same head
			NODE		* m_tail;
			NODE		* m_head;
			float		* m_D;				///< edge potential
			float		* m_msg;			///< message
		};
	};
}

