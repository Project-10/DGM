// Chain exact inference class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "MessagePassing.h"

namespace DirectGraphicalModels
{
	// ================================ Chain Infer Class ===============================
	/**
	* @ingroup moduleDecode
	* @brief Inference for chain graphs
	* @details Inference for Markov chains, based on the Chapman-Kolmogorov equations.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CInferChain : public CMessagePassing
	{
	public:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/
		DllExport CInferChain(CGraph * pGraph) : CMessagePassing(pGraph) {}
		DllExport virtual ~CInferChain(void) {}


	protected:
		/**
		* @brief Calculates messages for exact inference in a chain graph
		* @details This function estimates the marginal potentials for each graph node and stores them as node potentials.
		* @param nIt is not used
		*/
		DllExport virtual void calculateMessages(unsigned int nIt);
	};
}
