// Intensity feature extraction class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "ILocalFeatureExtractor.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Intensity Class ==============================
	/**
	* @brief Intensity feature extraction class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CIntensity : public ILocalFeatureExtractor
	{
	public: 
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC3.
		*/
		DllExport CIntensity(const Mat &img) : ILocalFeatureExtractor(img) {}
		DllExport virtual ~CIntensity(void) {}

		DllExport virtual Mat	get(void) const {return get(m_img);}
		
		/**
		* @brief Extracts the intesity feature.
		* @details This function calculates the intesity of the input image as follows: \f[ intensity=weight_0\cdot img.RED+weight_1\cdot img.GREEN+weight_2\cdot img.BLUE \f]
		* @param img Input image of type \b CV_8UC3.
		* @param weight The weight coefficients, which determine the contribution of each color channel to the resulting intensity.
		* @return The intesity feature image of type \b CV_8UC1.
		*/
		DllExport static Mat	get(const Mat &img, CvScalar weight = CV_RGB(0.333, 0.333, 0.333));
	};
} }