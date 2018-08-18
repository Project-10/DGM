// Viterbi inference class interface
// Writen by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "InferLBP.h"

namespace DirectGraphicalModels
{
	// ==================== Viterbi Infer Class ==================
	/**
	* @ingroup moduleDecode
	* @brief Max product Viterbi inference class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CInferViterbi : public CInferLBP
	{
	public:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/			
		DllExport CInferViterbi(CGraphPairwise *pGraph) : CInferLBP(pGraph) { m_maxSum = true; };
		DllExport virtual ~CInferViterbi(void) {};
	};

}
