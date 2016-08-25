// Interface class for feature extraction algorithms
// Writtem by Sergey G. Kosov in 2015 - 2016 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Base Feature Extractor Class ==============================
	/**
	* @brief Interface class for feature extraction algorithms
	* @ingroup moduleFEX
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/		
	class IFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image.
		*/
		DllExport IFeatureExtractor(const Mat &img) : m_img(img) {}
		DllExport virtual ~IFeatureExtractor(void) {}

		/**
		* @brief Extracts and returns the required feature.
		* @returns The feature image.
		*/
		DllExport virtual Mat	get(void) const = 0;


	protected:
		const Mat	m_img;		///< Container for the image, from which the features are to be extracted.
	};
} }