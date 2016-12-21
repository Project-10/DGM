// SIFT feature extraction class interface
// Written by Sergey G. Kosov in 2016 for Project X
#pragma once

#include "ILocalFeatureExtractor.h"

namespace DirectGraphicalModels { namespace fex
{

	
	// ================================ SIFT Class ==============================
	/**
	* @brief SIFT (<a href="https://en.wikipedia.org/wiki/Scale-invariant_feature_transform" target="_blank">scale-invariant feature transform</a>) feature extraction class.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/		
	class CSIFT : public ILocalFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		*/
		DllExport CSIFT(const Mat &img) : ILocalFeatureExtractor(img) {}
		DllExport virtual ~CSIFT(void) {}

		DllExport virtual Mat	get(void) const { return get(m_img); }

		/**
		* @brief Extracts the SIFT feature.
		*/
		DllExport static Mat	get(const Mat &img);
	};
} }