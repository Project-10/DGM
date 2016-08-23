// Variance feature extraction class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "BaseFeatureExtractor.h"
#include "SquareNeighborhood.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ HOG Class ==============================
	/**
	* @brief Variance feature extraction class.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/		
	class CVariance : public CBaseFeatureExtractor 
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		*/
		DllExport CVariance(const Mat &img) : CBaseFeatureExtractor(img) {}
		DllExport virtual ~CVariance(void) {}

		DllExport virtual Mat	get(void) const { return get(m_img); }

		/**
		* @brief Extracts the variance feature.
		* @details For each pixel of the source image this function calculates the variance within the pixel's neighbourhood \a nbhd.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		* @param nbhd Neighborhood around the pixel, where the variance is estimated. (Ref. @ref SqNeighbourhood).
		* @return The variance feature image of type \b CV_8UC1.
		*/
		DllExport static Mat	get(const Mat &img, SqNeighbourhood nbhd = sqNeighbourhood(5));

	};
} }
