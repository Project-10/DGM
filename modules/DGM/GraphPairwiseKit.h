// Kit class for constructing the Dense Pairwise objects
// Written by Sergey Kosov in 2018 - 2019 for Project X
#pragma once

#include "GraphKit.h"

#include "GraphPairwise.h"

#include "MessagePassing.h"
#include "InferLBP.h"
#include "InferTRW.h"
#include "InferViterbi.h"

#include "GraphPairwiseExt.h"

#include "macroses.h"

namespace DirectGraphicalModels
{
	/// Types of the inference / decoding objects
	enum class INFER { 
		LBP,		///< Loopy Belief Propagation inference
		TRW,		///< Convergent Tree-Reweighted inference
		Viterbi		///< Viterbi inference
	};

	// ================================ Pairwise Graph Kit Class ===============================
	/**
	* @brief Kit class for constructing Pairwise Graph objects
	* @ingroup moduleGraphKit
	* @author Dr. Sergey Kosov, sergey.kosov@project-10.de
	*/
	class CGraphPairwiseKit : public CGraphKit {
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		* @param infer
		*/	
		DllExport CGraphPairwiseKit(byte nStates, INFER infer = INFER::LBP)
			: CGraphKit()
			, m_graph(nStates)
			, m_graphExtension(m_graph)
		{
			switch (infer)
			{
			case INFER::LBP:	 m_pInfer = std::make_unique<CInferLBP>(m_graph); break;
			case INFER::TRW:	 m_pInfer = std::make_unique<CInferTRW>(m_graph); break;
			case INFER::Viterbi: m_pInfer = std::make_unique<CInferViterbi>(m_graph); break;
			default: DGM_ASSERT_MSG(false, "Unknown inference method");
			}
		}
		DllExport virtual ~CGraphPairwiseKit() = default;
 
		DllExport CGraph&		getGraph() override { return m_graph; }
		DllExport CInfer&		getInfer() override { return *m_pInfer; }
		DllExport CGraphExt&	getGraphExt() override { return m_graphExtension; }


	private:
		CGraphPairwise						m_graph;				///< Pairwise graph
		std::unique_ptr<CMessagePassing>	m_pInfer;				///< Inferer for pairwise graphs
		CGraphPairwiseExt					m_graphExtension;		///< Pairwise graph extension
	};
}