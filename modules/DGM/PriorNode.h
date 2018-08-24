// Node prior probability estimation class
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "Prior.h"

namespace DirectGraphicalModels
{
// ================================ Node Prior Class ================================
	/**
	* @brief %Node prior probability estimation class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CPriorNode : public CPrior
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		*/
		DllExport CPriorNode(byte nStates) : CBaseRandomModel(nStates), CPrior(nStates, RM_UNARY)  {}
		DllExport ~CPriorNode(void) {}

		/**
		* @brief Adds ground truth values to the co-occurance histogram vector
		* @param gt Matrix, each element of which is a ground-truth state (class)
		*/
		DllExport void	addNodeGroundTruth(const Mat &gt);
		/**
		* @brief Adds a ground truth value to the co-occurance histogram vector
		* @param gt The ground-truth state (class)
		*/
		DllExport void	addNodeGroundTruth(byte gt);


	protected:
		/**
		* @brief Calculates the prior probabilies.
		* @details This function returns the normalized class co-occurance histogram, which ought to be build during the training phase with help of the addNodeGroundTruth() function.
		* @return Prior node probability vector: Mat(size: nStates x 1; type: CV_32FC1)
		*/
		DllExport Mat	calculatePrior(void) const;
	};
}
