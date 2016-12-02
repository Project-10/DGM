// Multi-Scale feature extraction class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "IFeatureExtractor.h"
#include "SquareNeighborhood.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Scale Class ==============================
	/**
	* @brief Scale feature extraction class.
	* @details This class allow for multi-scale feature extraction.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/		
	class CScale : public ILocalFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		*/
		DllExport CScale(const Mat &img) : ILocalFeatureExtractor(img) {}
		DllExport virtual ~CScale(void) {}

		DllExport virtual Mat	get(void) const {return get(m_img);}

		/**
		* @brief Extracts the scale feature.
		* @details For each pixel of the source image this function calculates the mean value within the pixel's neighbourhood \a nbhd.
		* Using different neighbourhood radii, it alows for different scale representations of the features.
		* @param img Input image of type \b CV_8Uxx with arbitrary number of channels.
		* @param nbhd Neighborhood around the pixel, where the mean is estimated. (Ref. @ref SqNeighbourhood).
		* @return The scale feature image of the same type as input image.
		*/
		DllExport static Mat	get(const Mat &img, SqNeighbourhood nbhd = sqNeighbourhood(5));
	};

} }