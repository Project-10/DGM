// Saturation feature extraction class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "BaseFeatureExtractor.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Saturation Class ==============================
	/**
	* @brief Saturation feature extraction class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CSaturation :	public CBaseFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC3.
		*/
		DllExport CSaturation(Mat &img) : CBaseFeatureExtractor(img) {}
		DllExport virtual ~CSaturation(void) {}

		DllExport virtual Mat get(void) const {return get(m_img);}

		/**
		* @brief Extracts the saturation feature.
		* @details This function represents the nput image in HSV (hue-saturation-value) color model and returns the saturation channel.
		* @param img Input image of type \b CV_8UC3.
		* @return The saturation feature image of type \b CV_8UC1.
		*/
		DllExport static Mat get(const Mat &img);

	};

} }