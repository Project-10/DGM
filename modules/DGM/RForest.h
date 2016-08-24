// Random Forest Intermediate class interface (based on OpenCV)
// Written by Sergey G. Kosov in 2013 for CV
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	// =========================== Random Forest Train Class ===========================
	/**
	* @brief Random Forest class
	* @details The main purpose of this class is to overload the predict() function of the base OpenCV \b ml::RTrees() class.
	* This is done in order to produce the potential vector instead of the class value.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CRForest : public CvRTrees
	{
	public:
		/**
		* @brief Constuctor
		* @param nStates Number of states (classes)
		*/
		CRForest(byte nStates) : CvRTrees(), m_nStates(nStates) {}
		virtual ~CRForest(void) {}
		/**
		* @brief Calculates potentials give a feature vector
		* @param featureVector Multi-dimensinal sample point \f$\textbf{f}\f$: Mat(size: nFeatures x 1)
		* @return Potential vector: Mat(size: nStates x 1; type: CV_32FC1);
		*/
		virtual Mat predict(const Mat &featureVector);


	private:
		byte m_nStates;
	};
}
