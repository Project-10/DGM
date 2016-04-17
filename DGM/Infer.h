// Base abstract class for random model inference
// Written by Sergey G. Kosov in 2013 for Chronos Vision GmbH
// Adopted by Sergey G. Kosov in 2015 for Project X
// Expanded by Sergey G. Kosov in 2016 for Project X
#pragma once

#include "Decode.h"

namespace DirectGraphicalModels 
{
	class CGraph;

	// ================================ Infer Class ===============================
	/**
	* @ingroup moduleDecode
	* @brief Base abstract class for random model inference
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CInfer : protected CDecode
	{
	public:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/
		DllExport CInfer(CGraph *pGraph) : CDecode(pGraph) {};
		DllExport virtual ~CInfer(void) {};

		/**
		* @brief Approximate decoding
		* @details This function calls first inference @ref infer() and then, using resulting marginal probabilities, estimates the most
		* probable configuration of states (classes) in the graph via CDecode::decode().
		* > This function modifies Node::Pot containers of graph nodes
		* @param nIt Number of iterations
		* @param lossMatrix (optional) The loss matrix \f$L\f$ (size: nStates x nStates; type: CV_32FC1).
		* It must be a quadratic zero-diagonal matrix, whith all non-diagonal elements \f$L_{i,j} > 0, \forall i\neq j\f$.
		* The elemets \f$L_{i,j}\f$ represent a loss if state \f$j\f$ is classified as a state \f$i\f$.
		* @return The most probable configuration
		* @note This function estimates the most likely configuration, based on the \a marginal probabilities (potentials) in graph nodes, which in
		* general is \b NOT the same as the set of most likely states, which corresponds to the configuration with the highest \a joint probability. 
		* In other words:
		* @code
		*	using namespace DirectGraphicalModels;
		*	CGraph  * graph   = new CGraph(nStates);
		*	CInfer  * inferer = new CInferExact(graph);
		*	CDecode * decoder = new CDecodeExact(graph);
		*	inferer->decode() == decoder->decode();		// This statement is not always true!
		* @endcode
		*/
		DllExport virtual byte	* decode(unsigned int nIt = 0, Mat &lossMatrix = Mat()) {
			if (nIt) infer(nIt); 
			return CDecode::decode(m_pGraph, lossMatrix);
		}
		/**
		* @brief Inference
		* @details This function estimates the marginal potentials for each graph node, and stores them as node potentials
		* > This function modifies Node::Pot containers of graph nodes
		* @param nIt Number of iterations
		* @note This function must not to be linear, \a i.e. \f$ infer(\alpha\times N)\not\equiv\alpha\times infer(N) \f$
		* @note This function substitutes the graph nodes' potentials with estimated marginal potentials
		*/
		DllExport virtual void	  infer(unsigned int nIt = 1) = 0;
		/**
		* @brief Returns the confidence of the prediction
		* @details This function calculates the confidence values for the predicted states (classes) in the graph via CInfer::decode().
		* The confidence values lie in range [0; 1].
		* @return The confidence values for each node of graph.
		*/
		DllExport vec_float_t	  getConfidence(void);
	};
}