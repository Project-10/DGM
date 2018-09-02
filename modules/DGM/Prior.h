// Base abstract class for prior probability estimation
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "BaseRandomModel.h"

namespace DirectGraphicalModels
{
	// ================================ Prior Class ================================
	/**
	* @brief Base abstract class for prior probability estimation.
	* @details This class implements serialization interface and defines basic interface for calculation prior probabilities.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CPrior : public virtual CBaseRandomModel
	{
	public:	
		/**
		* @brief Constructor.
		* @param nStates Number of states (classes).
		* @param type Type of the random model (Ref. @ref RandomModelType)
		*/
		DllExport CPrior(byte nStates, RandomModelType type);
		DllExport ~CPrior(void);

		DllExport void			reset(void);

		/**
		* @brief Returns the prior probabilies.
		* @details This function calls calculatePrior() function, which should be implemented in derived classes. It returns the normalized class co-occurance histogram,
		* multiplied with the parameter \b weight. If the prior probabilities were not estimated, this functions returns a uniform distribution "all ones".
		* @param weight The weighting parameter
		* @returns 1D (nStates) for node, 2D (nStates x nStates) for edge or 3D (nStates x nStates x nStates) for triplet Mat of type CV_32FC1 with prior probabilies.
		*/
		DllExport Mat			getPrior(float weight = 1.0f) const;

	
	protected:
		DllExport virtual void	saveFile(FILE *pFile) const;
		DllExport virtual void	loadFile(FILE *pFile);		
		/**
		* @brief Calculates the prior probabilies.
		* @details This function returns the normalized class co-occurance histogram, which ought to be build during the training phase with help of the "addGroundTruth()" functionality, 
		* implemented in derived classes. 
		* @returns 1D (nStates) for node, 2D (nStates x nStates) for edge or 3D (nStates x nStates x nStates) for triplet Mat of type CV_32FC1 with prior probabilies.		
		*/
		DllExport virtual Mat		calculatePrior(void) const = 0;


	protected:
		Mat						m_histogramPrior;		///< The class cooccurance histogram
		
	
	private:		
		RandomModelType			m_type;					///< Type of the random model (@ref RandomModelType)
	};
}