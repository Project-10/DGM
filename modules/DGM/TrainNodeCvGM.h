// Gaussian Model (based on OpenCV) training class interface
// Written by Sergey G. Kosov in 2012 for Project X
#pragma once

#include "TrainNodeCvGMM.h"

namespace DirectGraphicalModels
{
	class CNDGauss;
	// ======================== OpenCV GM Train Class =========================
	/**
	* @ingroup moduleTrainNode
	* @brief OpenCV Gaussian Model training class
	* @details This class realizes the generative training mechanism, based on the idea of approximating a the density of multi-dimensional random variables 
	* with a single multi-dimensional Gaussian function.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeCvGM : public CTrainNodeCvGMM
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		*/
		DllExport CTrainNodeCvGM(byte nStates, word nFeatures) : CTrainNodeCvGMM(nStates, nFeatures, 0, 1), CBaseRandomModel(nStates) {}
		DllExport ~CTrainNodeCvGM(void) {}
	};
}