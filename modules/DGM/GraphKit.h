// Abstract Kit classes for constructing the Graph instances
// Written by Sergey Kosov in 2018 for Project X
#pragma once

#include "Graph.h"
#include "GraphPairwise.h"

#include "Infer.h"
#include "MessagePassing.h"
#include "InferLBP.h"
#include "InferTRW.h"
#include "InferViterbi.h"
#include "InferDense.h"

#include "GraphExt.h"
#include "GraphPairwiseExt.h"
#include "GraphDenseExt.h"

#include "macroses.h"

namespace DirectGraphicalModels 
{
	// ================================ Graph Kit Abstract Factory Class ===============================
	/**
	* @brief Abstract Kit class for constructing Graph objects
	* @ingroup moduleGraphKit
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphKit {
	public:
		CGraphKit() = default;
		CGraphKit(const CGraphKit&) = delete;
		virtual ~CGraphKit() = default;
		const CGraphKit& operator= (const CGraphKit&) = delete;

		/**
		* @brief 
		*/
		virtual CGraph&		getGraph() = 0;
		/**
		* @brief
		*/
		virtual CInfer&		getInfer() = 0;
		/**
		* @brief
		*/
		virtual CGraphExt&	getGraphExt() = 0;
	};

	enum class INFER {LBP, TRW, Viterbi};

	// ================================ Pairwise Graph Kit Class ===============================
	/**
	* @brief Kit class for constructing Pairwise Graph objects
	* @ingroup moduleGraphKit
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphPairwiseKit : public CGraphKit {
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		* @param infer
		*/	
		CGraphPairwiseKit(byte nStates, INFER infer = INFER::LBP)
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
		virtual ~CGraphPairwiseKit() {}
 
		virtual CGraph&		getGraph() { return m_graph; }
		virtual CInfer&		getInfer() { return *m_pInfer; }
		virtual CGraphExt&	getGraphExt() { return m_graphExtension; }


	private:
		CGraphPairwise						m_graph;
		std::unique_ptr<CMessagePassing>	m_pInfer;
		CGraphPairwiseExt					m_graphExtension;
	};


	// ================================ Dense Graph Kit Class ===============================
	/**
	* @brief Kit class for constructing Dense Graph objects
	* @ingroup moduleGraphKit
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraphDenseKit : public CGraphKit {
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		CGraphDenseKit(byte nStates)
			: CGraphKit()
			, m_graph(nStates)
			, m_infer(m_graph)
			, m_graphExtension(m_graph)
		{}
		virtual ~CGraphDenseKit() {}

		virtual CGraph&		getGraph() { return m_graph; }
		virtual CInfer&		getInfer() { return m_infer; }
		virtual CGraphExt&	getGraphExt() { return m_graphExtension; }


	private:
		CGraphDense		m_graph;
		CInferDense		m_infer;
		CGraphDenseExt	m_graphExtension;
	};

}