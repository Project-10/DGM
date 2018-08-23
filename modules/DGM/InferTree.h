// Tree and DAG exact inference class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "MessagePassing.h"

namespace DirectGraphicalModels 
{
	// ==================== Tree Infer Class ==================
	/**
	* @ingroup moduleDecode
	* @brief Inference for tree graphs (undirected graphs without loops)
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	* @todo Check the application of this class to DAGs and mixed graphs
	*/
	class CInferTree : public CMessagePassing
	{
	public:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/
		DllExport CInferTree(CGraphPairwise * pGraph) : CMessagePassing(pGraph) {}
		DllExport virtual ~CInferTree(void) {}


	protected:
		/**
		* @brief Calculates messages for exact inference in a tree graph
		* @details This function estimates the marginal potentials for each graph node and stores them as node potentials.
		* @param nIt is not used
		*/
		DllExport virtual void calculateMessages(unsigned int nIt);
	};
}
