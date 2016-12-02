// Hue, Saturation and Value feature extraction class interface
// Written by Sergey G. Kosov in 2016 for Project X
#pragma once

#include "IFeatureExtractor.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Saturation Class ==============================
	/**
	* @brief Hue, Saturation and Value feature extraction class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CHSV : public ILocalFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC3.
		*/
		DllExport CHSV(Mat &img) : ILocalFeatureExtractor(img) {}
		DllExport virtual ~CHSV(void) {}

		DllExport virtual Mat get(void) const { return get(m_img); }

		/**
		* @brief Extracts the HSV feature.
		* @details This function transforms the input image into HSV (hue-saturation-value) color space.
		* @param img Input image of type \b CV_8UC3.
		* @return The (hue-saturation-value) feature image of type \b CV_8UC3.
		*/
		DllExport static Mat get(const Mat &img);
	};
} }