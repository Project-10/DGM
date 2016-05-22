#pragma once

#include "MessagePassing.h"

namespace DirectGraphicalModels
{
	class CInferTRW : public CMessagePassing
	{
	public:
		DllExport CInferTRW(CGraph *pGraph) : CMessagePassing(pGraph) {}
		DllExport virtual ~CInferTRW(void) {}

		DllExport virtual void	  infer(unsigned int nIt = 1);


	protected:
		virtual void calculateMessages(unsigned int nIt) {}


	private:
		void	  TransformPotentials(void);													// Changes the potentials; pot = -log(pot)
		void	  TransformPotentialsBack(void);												// Changes the potentials; pot = exp(-pot)


		/******************************************************************
		Vladimir Kolmogorov, 2005
		(c) Microsoft Corporation. All rights reserved.
		*******************************************************************/
		void	  ComputeSolution(void); 														// sets Node::m_solution
		float	  UpdateMessage(Edge *edge, float *source, float gamma, int dir, float *buf);

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


	};
}
