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
	* @brief Dense Inference for Dense CRF. 
	* @details The implementation is based on 
	* <a href="http://graphics.stanford.edu/projects/densecrf/densecrf.pdf">Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials</a> paper. 
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CInferDense : public CInfer 
	{
	public:
		/**
		* @brief Constructor
		* @param graph The graph
		*/
		DllExport CInferDense(CGraphDense& graph) : CInfer(graph) {}
		DllExport virtual ~CInferDense(void) = default;
	
		DllExport virtual void	infer(unsigned int nIt = 1);


	protected:
		/**
		* @brief Returns the dense graph
		* @return The dense graph
		*/
		CGraphDense& getGraphDense(void) const { return dynamic_cast<CGraphDense&>(getGraph()); }
	};
}
