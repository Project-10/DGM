// Interface class for random model training
// Written by Sergey G. Kosov in 2012 for Project X
#pragma once

#include "BaseRandomModel.h"

namespace DirectGraphicalModels
{
	// ================================ Train Class ================================
	/**
	* @brief Interface class for random model training
	* @ingroup moduleTrain
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class ITrain : public virtual CBaseRandomModel
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		*/
		DllExport ITrain(byte nStates, word nFeatures) : CBaseRandomModel(nStates), m_nFeatures(nFeatures) {}
		DllExport virtual ~ITrain(void) {}

		/**
		* @brief Random model training
		* @details Auxilary function for training - some derived classes may use this function inbetween training and classification phases
		* @note This function \b must be called inbetween the training and classification phases
		* @param doClean Flag indicating if the memory, keeping the trining data should be released after training
		*/		
		DllExport virtual void	train(bool doClean = false) = 0;	
		/**
		* @brief Returns number of features
		* @return Number of features @ref m_nFeatures
		*/		
		DllExport word			getNumFeatures(void) const { return m_nFeatures; }


	protected:
		word	m_nFeatures;						///< The number of features (length of the feature vector)
	};



}

