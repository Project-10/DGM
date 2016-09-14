// Triplet prior probability estimation class
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "Prior.h"

namespace DirectGraphicalModels
{
// ================================ Edge Prior Class ================================
/**
@brief %Triplet prior probability estimation class.
@author Sergey G. Kosov, sergey.kosov@project-10.de
*/	
	class CPriorTriplet : public CPrior
	{
	public:
/**
@brief Constructor
@param nStates Number of states (classes).
*/
		DllExport CPriorTriplet(byte nStates) : CPrior(nStates, RM_TRIPLET), CBaseRandomModel(nStates) {}
		DllExport ~CPriorTriplet(void) {}

/**
@brief Adds the groud-truth value to the co-occurance histogram matrix
@details Here \b gt1 is the X-coordinate of the co-occurance histogram matrix, \b gt2 - Y-coordinate of the co-occurance histogram matrix and \b gt3 - Z-coordinate of the co-occurance histogram matrix.
@param gt1 The ground-truth state (value) of the first node in triplet. 
@param gt2 The ground-truth state (value) of the second node in triplet. 
@param gt3 The ground-truth state (value) of the third node in triplet. 
*/
		DllExport void	addTripletGroundTruth(byte gt1, byte gt2, byte gt3); 


	protected:
		/**
		* @brief Calculates the prior probabilies.
		* @details This function returns the normalized class co-occurance histogram, which ought to be build during the training phase with help of the addTripletGroundTruth() function.
		* @return Prior edge probability voxel: Mat(size: nStates x nStates x nStates; type: CV_32FC1)
		*/
		DllExport Mat	calculatePrior(void) const;

	};	

}