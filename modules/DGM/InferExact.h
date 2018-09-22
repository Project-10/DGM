// Exact inference class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "Infer.h"
#include "DecodeExact.h"

namespace DirectGraphicalModels
{
	// ================================ Exact Infer Class ===============================
	/**
	* @ingroup moduleDecode
	* @brief Exact inference class
	* @note Use this class only if \f$ nStates^{nNodes} < 2^{32}\f$
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CInferExact : public CInfer, private CDecodeExact
	{
	public:
		/**
		* @brief Constructor
		* @param graph The graph
		*/
		DllExport CInferExact(CGraphPairwise &graph) : CInfer(graph), CDecodeExact(graph) {}
		DllExport virtual ~CInferExact(void) {}

		/**
		* @brief Exact inference
		* @details This function estimates the most likely configuration, based on the marginal probabilities (potentials) in graph nodes, which in
		* general is \b NOT the same as the set of most likely states. It means the following:
		* @code
		* using namespace DirectGraphicalModels;
		* CGraphPairwise  * graph   = new CGraphPairwise(nStates);
		* CInfer  * inferer = new CInferExact(graph);
		* CDecode * decoder = new CDecodeExact(graph);
		* inferer->decode() == decoder->decode();		// This statement is not always true!
		* @endcode
		* @param nIt is not used 
		* @note This function changes the graph nodes' potentials
		*/
		DllExport virtual void	  infer(unsigned int nIt = 0);
        
        using CInfer::decode;
	};
}
