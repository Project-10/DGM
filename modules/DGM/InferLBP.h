// Loopy Belief Propagation inference class interface
// Written by Sergey G. Kosov in 2013 for Chronos Vision GmbH
// Adopted by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "MessagePassing.h"

namespace DirectGraphicalModels
{
	// ==================== Loopy Belief Propagation Infer Class ==================
	/**
	* @ingroup moduleDecode
	* @brief Sum product Loopy Belief Propagation inference class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CInferLBP : public CMessagePassing
	{
	public:
		/**
		* @brief Constructor
		* @param graph The graph
		*/			
		DllExport CInferLBP(CGraphPairwise &graph) : CMessagePassing(graph), m_maxSum(false) {}
		DllExport virtual ~CInferLBP(void) = default;


	protected:
		DllExport virtual void	calculateMessages(unsigned int nIt);
		void					setMaxSum(bool maxSum) { m_maxSum = maxSum; }
		bool					isMaxSum(void) const { return m_maxSum; }


	private:
		bool m_maxSum;			///< Flag indicating weather the max-sum LBP (Viterbi algorithm) should be applied
	};

}
