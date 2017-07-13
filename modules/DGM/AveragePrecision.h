// Average Precision class
// Written by Sergey Kosov for Project X in 2016
#pragma once

#include "types.h"

namespace DirectGraphicalModels 
{
	/**
	* @ingroup moduleEva
	* @brief Returns <a href="https://www.youtube.com/watch?v=pM6DJ0ZZee0" target="_blank">Average Precision</a> for the selected state (class) \b state
	* @details This function analyses 3 values of every image pixel, namely the <i>predicted state</i>, the <i>groundtruth state</i> and
	* the <i>potential</i> of the pixel to be the class \b state, passed to the function via the arguments: \b predictions, \b gt and
	* \b potentials, respectively. First, the pixels are sorted by the \a potentials in descending order. After that, the following algorithm is applied:
	* @code
	* for (int i = 1; i <= numPixels; i++) 
	*	if (Pixel[i].groundtruth_state == state) {				
	*		numRelevants++;
	*		if (Pixel[i].predicted_state == state) {			
	*			numCoincidences++;
	*			AP = AP + numCoincidences / i;
	*		}
	*	}
	* AP = AP / nRelevants;
	* @endcode
	* where AP stays for Average Precision.
	* @param predictions The most probable configuration, returned by the CDecode::decode() function
	* @param potentials The potential values for each node of the graph, returned by the CInfer::getPotentials(\b state) function
	* @param gt The groundtruth values for each node of the graph.
	* May be converted from a groundtruth image as follows:
	* @code
	* vec_byte_t gt(gtImg.data, gtImg.data + gtImg.cols * gtImg.rows);
	* @endcode
	* @param state The state (class) for which the Average Precision is calculated
	* @returns The Average Precision value
	*/	
	DllExport float getAveragePrecision(const vec_byte_t &predictions, const vec_float_t &potentials, const vec_byte_t &gt, byte state);

}