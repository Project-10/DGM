// Kit class for constructing the Dense Graph objects
// Written by Sergey Kosov in 2018 - 2019 for Project X
#pragma once

#include "GraphKit.h"
#include "InferDense.h"
#include "GraphDenseExt.h"

namespace DirectGraphicalModels
{
	// ================================ Dense Graph Kit Class ===============================
	/**
	* @brief Kit class for constructing Dense Graph objects
	* @ingroup moduleGraphKit
	* @author Dr. Sergey Kosov, sergey.kosov@project-10.de
	*/
	class CGraphDenseKit : public CGraphKit {
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport CGraphDenseKit(byte nStates)
			: CGraphKit()
			, m_graph(nStates)
			, m_infer(m_graph)
			, m_graphExtension(m_graph)
		{}
		DllExport virtual ~CGraphDenseKit() = default;

		DllExport CGraph&		getGraph() override { return m_graph; }
		DllExport CInfer&		getInfer() override { return m_infer; }
		DllExport CGraphExt&	getGraphExt() override { return m_graphExtension; }


	private:
		CGraphDense		m_graph;			///< Dense (complete) graph
		CInferDense		m_infer;			///< Inferer for dense graphs
		CGraphDenseExt	m_graphExtension;	///< Dense graph extension
	};
}