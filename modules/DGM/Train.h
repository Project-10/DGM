// Base abstract class for random model training
// Written by Sergey G. Kosov in 2012 for Project X
#pragma once

#include "BaseRandomModel.h"

namespace DirectGraphicalModels
{
	// ================================ Train Class ================================
	/**
	* @brief Base abstract class for random model training
	* @ingroup moduleTrain
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrain : public virtual CBaseRandomModel
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		*/
		DllExport CTrain(byte nStates, word nFeatures) : CBaseRandomModel(nStates), m_nFeatures(nFeatures) {}
		DllExport virtual ~CTrain(void) {}

		/**
		* @brief Random model training
		* @details Auxilary function for training - some derived classes may use this function inbetween training and classification phases
		* @note This function \b must be called inbetween the training and classification phases
		*/		
		DllExport virtual void	train(void) = 0;	
		/**
		* @brief Returns number of features
		* @return Number of features @ref m_nFeatures
		*/		
		DllExport word			getNumFeatures(void) const {return m_nFeatures;}


	protected:
		word	m_nFeatures;						///< The number of features (length of the feature vector)
	};



}

