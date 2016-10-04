// Mean Average Precision class
// Written by Sergey Kosov for Project X in 2016
#pragma once

#include "types.h"

namespace DirectGraphicalModels 
{
	/**
	* @ingroup moduleEva
	* @brief Returns <a href="https://www.youtube.com/watch?v=pM6DJ0ZZee0" target="_blank">Mean Average Precision (MAP)</a> for the selected state (class) \b gt
	* @param predictions The most probable configuration, returned by the CDecode::decode() function
	* @param potentials potential values for each node of the graph, returned by the CInfer::getPotentials() function
	* @param gt The groundtruth state (class)
	* @returns The MAP value
	*/	
	DllExport float getMAP(const vec_byte_t &predictions, const vec_float_t &potentials, byte gt);

}