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
		* @param graph The graph
		*/			
		DllExport CInferViterbi(CGraphPairwise &graph) : CInferLBP(graph) { setMaxSum(true); }
		DllExport virtual ~CInferViterbi(void) {};
	};

}
