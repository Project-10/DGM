// Global-features Extraction class 
// Writtem by Sergey G. Kosov in 2016 for Project X
#pragma once

#include "IFeatureExtractor.h"
#include "Global.h"

namespace DirectGraphicalModels { namespace fex 
{
	// ================================ Base GLOBAL Feature Extractor Class ==============================
	/**
	* @ingroup moduleGFEX
	* @brief Interface class for global feature extraction algorithms
	* @details The derived algorithms are supposed to extract one feature per image
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGlobalFeatureExtractor : public IFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image.
		*/
		DllExport CGlobalFeatureExtractor(const Mat &img) : IFeatureExtractor(img) { }
		DllExport virtual ~CGlobalFeatureExtractor(void) { }

		/**
		* @brief Returns the number of staight lines in the image.
		* @param threshold1 The higher threshold of the two, passed to the \a Canny edge detector(the lower one is twice smaller).
		* @param threshold2 The accumulator threshold parameter.Only those lines are detected that get enough votes(> \b threshold2).
		* @returns The number of staight lines in the source image
		*/
		DllExport size_t getNumLines(double threshold1 = 100, double threshold2 = 50) const { return global::getNumLines(m_img, threshold1, threshold2); }
		/**
		* @brief Returns the number of circles in the source image
		* @param threshold1 The higher threshold of the two, passed to the \a Canny edge detector (the lower one is twice smaller).
		* @param threshold2 The accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. 
		* @return The number of circles in the source image
		*/
		DllExport size_t getNumCircles(double threshold1 = 100, double threshold2 = 30) const { return global::getNumCircles(m_img, threshold1, threshold2); }
	};

} }