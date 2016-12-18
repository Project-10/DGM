// Interface class for local-feature extraction algorithms
// Writtem by Sergey G. Kosov in 2016 for Project X
#pragma once

#include "IFeatureExtractor.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ Base LOCAL Feature Extractor Class ==============================
	/**
	* @ingroup moduleLFEX
	* @brief Interface class for local feature extraction algorithms
	* @details The derived algorithms are supposed to extract features for each image pixel
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class ILocalFeatureExtractor : public IFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image.
		*/
		ILocalFeatureExtractor(const Mat &img) : IFeatureExtractor(img) {}
		virtual ~ILocalFeatureExtractor(void) {}

		/**
		* @brief Extracts and returns the required feature.
		* @returns The feature image.
		*/
		virtual Mat	get(void) const = 0;
	};
} }