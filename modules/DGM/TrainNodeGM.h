// Gaussian Model training class interface
// Written by Sergey G. Kosov in 2012 for Project X
#pragma once

#include "TrainNodeGMM.h"

namespace DirectGraphicalModels
{
	class CNDGauss;
	// ======================== Gaussian Model Train Class =========================
	/**
	* @ingroup moduleTrainNode
	* @brief Gaussian Model training class
	* @details This class realizes the generative training mechanism, based on the idea of approximating a the density of multi-dimensional random variables 
	* with a single multi-dimensional Gaussian function.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeGM : public CTrainNodeGMM
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		*/
		DllExport CTrainNodeGM(byte nStates, word nFeatures) : CBaseRandomModel(nStates), CTrainNodeGMM(nStates, nFeatures, 1) {}
		DllExport ~CTrainNodeGM(void) = default;

		/**
		* @brief Returns the node potential, based on the feature vector
		* @details This function calculates the potentials of the node, described with the sample \a featureVector (\f$ \textbf{f} \f$):
		* \f$ nodePot_s = \mathcal{N}_s(\textbf{f}), \forall s \in \mathbb{S} \f$, where \f$\mathbb{S}\f$ is the set of all states (classes). 
		* In other words, the indexes: \f$ s \in [0; nStates) \f$. Here \f$ \mathcal{N} \f$ is a Gaussian function kernel, described in class @ref CKDGauss
		* @param featureVector Multi-dimensinal point \f$\textbf{f}\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1)
		* @param weight The weighting parameter
		* @return %Node potentials on success: Mat(size: nStates x 1; type: CV_32FC1); empty Mat() otherwise
		*/		
		DllExport Mat getNodePotentials(const Mat &featureVector, float weight = 1.0f) {return CTrainNodeGMM::getNodePotentials(featureVector, weight);}	// This is done for doxygen comment
	};
}
