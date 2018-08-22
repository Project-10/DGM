// Base abstract class for random model decoding
// Written by Sergey G. Kosov in 2013-2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	class CGraph;

	// ================================ Decode Class ===============================
	/**
	* @ingroup moduleDecode
	* @brief Base abstract class for random model decoding
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CDecode
	{
	protected:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/
		DllExport CDecode(CGraph *pGraph) : m_pGraph(pGraph) {};
		

	public:
		DllExport virtual ~CDecode(void) {};
		/**
		* @brief Approximate decoding
		* @details This function estimates the most probable configuration of states (classes) in the graph,
		* based on marginal probabilities in graph nodes.
		* @param nIt Number of iterations
		* @param lossMatrix (optional) The loss matrix \f$L\f$ (size: nStates x nStates; type: CV_32FC1). 
		* It must be a quadratic zero-diagonal matrix, whith all non-diagonal elements \f$L_{i,j} > 0, \forall i\neq j\f$.
		* The elemets \f$L_{i,j}\f$ represent a loss if state \f$j\f$ is classified as a state \f$i\f$.
		* @return The most probable configuration
		*/
		DllExport virtual vec_byte_t decode(unsigned int nIt = 0, Mat &lossMatrix = EmptyMat) const { return decode(m_pGraph, lossMatrix); }
		/**
		* @brief Approximate decoding
		* @details This function estimates the most probable configuration of states (classes) in the graph,
		* based on marginal probabilities in graph nodes.
		* @param pGraph The graph
		* @param lossMatrix (optional) The loss matrix \f$L\f$ (size: nStates x nStates; type: CV_32FC1). 
		* It must be a quadratic zero-diagonal matrix, whith all non-diagonal elements \f$L_{i,j} > 0, \forall i\neq j\f$.
		* The elemets \f$L_{i,j}\f$ represent a loss if state \f$j\f$ is classified as a state \f$i\f$.
		* @return The most probable configuration
		*/
		DllExport static vec_byte_t	decode(const CGraph *pGraph, Mat &lossMatrix = EmptyMat);
		/**
		* @brief Returns a default loss matrix \f$L\f$
		* @param nStates The number of States (classes)
		* @return a loss matrix \f$nStates\times nStates\f$: \f$L=\left\{\begin{array}{rl}0&\mbox{if i = j}\\ 1&\mbox{otherwise}\end{array}\right.\f$
		* (size: nStates x nStates; type: CV_32FC1). 
		* @note Resulting loss matrix will cause no effect when using inside the decode() function. 
		* This function provides only a default matrix for further user modification before using in the decode() function.
		*/
		DllExport static Mat	  getDefaultLossMatrix(byte nStates);


	protected:
		/**
		* @brief Returns the pointer to the graph
		* @return The pointer to the graph
		*/
		CGraph * getGraph(void) const { return m_pGraph; }


	private:
		CGraph * m_pGraph;		///< Pointer to the graph


	private:
		// Copy semantics are disabled
		CDecode(const CDecode &rhs) {}
		const CDecode & operator= (const CDecode & rhs) {return *this;}
	};
}
