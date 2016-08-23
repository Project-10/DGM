// Probability Density Function abstract class interface
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "BaseRandomModel.h"

namespace DirectGraphicalModels 
{
// ================================ PDF Class ==============================
	/**
	* @brief Probability Density Function (PDF) abstract class
	* @details This class define the interface for estimation of 1D probability density functions for 
	* random read-valued variables
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CPDF : public CBaseRandomModel 
	{
	friend class CTrainNodeNaiveBayes;
	
	public:
		DllExport CPDF(void) : CBaseRandomModel(0), m_nPoints(0) {}
		DllExport virtual ~CPDF(void) {}

		/**
		* @brief Adds a sample point for PDF estimation.
		* @param point The sample point.
		*/
		DllExport virtual void	addPoint(float point) = 0;
		/**
		* @brief Returns the probability density value for the argument \b point.
		* @param point The sample point.
		* @returns The corresponding probaility density value.
		*/
		DllExport virtual float	getDensity(float point) = 0;
		/**
		* @brief Returns the lower argument boundary of the PDF
		* @returns The lower bound
		*/
		DllExport virtual float	min(void) const = 0;
		/**
		* @brief Returns the upper argument boundary of the PDF
		* @returns The upper bound
		*/
		DllExport virtual float	max(void) const = 0;
		/**
		* @brief Checks weather the PDF was estimated.
		* @retval true if at least one sample was added with the addPoint() function.
		* @retval false otherwise
		*/
		DllExport bool			isEstimated(void) { return m_nPoints != 0; }


	protected:
		long	m_nPoints;				///< The number of samples, added with the addPoint() function

	
	};
}