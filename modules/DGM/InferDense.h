// Dense inference class interface
// Written by Sergey G. Kosov in 2018 for Project X (insired by Philipp Kraehenbühl Dense CRF implementation) 
#pragma once

#include "Infer.h"
#include "GraphDense.h"

namespace DirectGraphicalModels
{
	// ================================ Infer Class ===============================
	/**
	* @ingroup moduleDecode
	* @brief Dense Inference for Dense CRF 
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CInferDense : public CInfer 
	{
	public:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/
		DllExport CInferDense(CGraphDense *pGraph) : CInfer(pGraph) {}
		DllExport virtual ~CInferDense(void) {}
	
		DllExport virtual void	infer(unsigned int nIt = 1);


	protected:
		/**
		* @brief Returns the pointer to the graph
		* @return The pointer to the graph
		*/
		CGraphDense * getGraphDense(void) const { return dynamic_cast<CGraphDense *>(getGraph()); }
	};
}