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
		IFeatureExtractor(const Mat &img) : m_img(img) {}
		virtual ~IFeatureExtractor(void) {}


	protected:
		const Mat	m_img;		///< Container for the image, from which the features are to be extracted.
	};

	
	// ================================ Base LOCAL Feature Extractor Class ==============================
	/**
	* @brief Interface class for local feature extraction algorithms
	* @details The derived algorithms are supposed to extract features for each image pixel
	* @ingroup moduleFEX
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

	
	// ================================ Base GLOBAL Feature Extractor Class ==============================
	/**
	* @brief Interface class for global feature extraction algorithms
	* @details The derived algorithms are supposed to extract one feature per image
	* @ingroup moduleFEX
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class IGlobaFeatureExtractor : public IFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image.
		*/
		IGlobaFeatureExtractor(const Mat &img) : IFeatureExtractor(img) {}
		virtual ~IGlobaFeatureExtractor(void) {}

		virtual float get(void) const = 0;
	};
} }