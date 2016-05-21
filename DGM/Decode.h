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
		DllExport virtual vec_byte_t decode(unsigned int nIt = 0, Mat &lossMatrix = Mat()) /*const*/ { return decode(m_pGraph, lossMatrix); }
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
		DllExport static vec_byte_t	decode(const CGraph *pGraph, Mat &lossMatrix = Mat());
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
		CGraph	* m_pGraph;		///< Pointer to the graph


	protected:
		/**
		* @brief Sets the \a state according to the configuration index \a configuration 
		* @details This function is used in exact inference / decoding
		* @param state Array of \a nNodes elements with the current configuration (states destributed along the nodes)
		* @param configuration Configuration index \f$\in[0; nStates^{nNodes}]\f$
		*/
		void		setState(vec_byte_t &state, qword configuration) const;
		/**
		* @brief Increases the \a state by one, \a i.e. switches the \a state array to the consequent configuration
		* @details This function is used in exact inference / decoding
		* @param state Array of \a nNodes elements with the current configuration (states destributed along the nodes)
		*/
		void		incState(vec_byte_t &state) const;
		/**
		* @brief Calculates potentials for all possible configurations
		* @details This function is used in exact inference / decoding
		* @return \f$nStates^{nNodes}\f$ potentials, corresponding to the all possible configurations (states destributed along the nodes)
		*/
		vec_float_t	calculatePotentials(void) const;


	
	private:
		// Copy semantics are disabled
		CDecode(const CDecode &rhs) {}
		const CDecode & operator= (const CDecode & rhs) {return *this;}
	};
}
