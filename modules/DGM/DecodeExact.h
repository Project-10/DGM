// Exact decoding class interface
// Written by Sergey G. Kosov in 2012 for Project X
#pragma once

#include "Decode.h"
#include "GraphPairwise.h"

namespace DirectGraphicalModels
{
	// ============================= Exact Decode Class ============================
	/**
	* @ingroup moduleDecode
	* @brief Exact decoding class
	* @note Use this class only if \f$ nStates^{nNodes} < 2^{32}\f$
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CDecodeExact : public CDecode
	{
	public:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/		
		DllExport CDecodeExact(CGraphPairwise *pGraph) : CDecode(pGraph) {}
		DllExport virtual ~CDecodeExact(void) {}

		/**
		* @brief Exact decoding
		* @param nIt is not used
		* @param lossMatrix is not used
		* @return The most probable configuration
		*/
		DllExport virtual vec_byte_t decode(unsigned int nIt = 0, Mat &lossMatrix = EmptyMat) const;


	protected:
		/**
		* @brief Returns the pointer to the graph
		* @return The pointer to the graph
		*/
		CGraphPairwise * getGraphPairwise(void) const { return reinterpret_cast<CGraphPairwise *>(CDecode::getGraph()); }
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
	};
}


