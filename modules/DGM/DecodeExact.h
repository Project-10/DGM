// Exact decoding class interface
// Written by Sergey G. Kosov in 2012 for Project X
#pragma once

#include "Decode.h"

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
		DllExport CDecodeExact(CGraph *pGraph) : CDecode(pGraph) {}
		DllExport virtual ~CDecodeExact(void) {}

		/**
		* @brief Exact decoding
		* @param nIt is not used
		* @param lossMatrix is not used
		* @return The most probable configuration
		*/
		DllExport virtual vec_byte_t decode(unsigned int nIt = 0, Mat &lossMatrix = EmptyMat) const;


	};
}


