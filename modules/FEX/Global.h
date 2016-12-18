//
// Writtem by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels { namespace fex {

	// ================================ Global Namespace ==============================
	/**
	* @ingroup moduleGFEX
	* @brief Global-features extraction
	* @details This namespace collects methods for the global features extraction
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	namespace global {
		/**
		* @brief Returns the number of staight lines in the image.
		* @param img The source image of type \b CV_8UC1 or \b CV_8UC3.
		* @param threshold1 The higher threshold of the two, passed to the \a Canny edge detector (the lower one is twice smaller).
		* @param threshold2 The accumulator threshold parameter. Only those lines are detected that get enough votes ( > \b threshold2).
		* @returns The number of staight lines in the source image
		*/
		DllExport size_t	getNumLines(const Mat &img, double threshold1 = 100, double threshold2 = 50);
		/**
		* @brief Returns the number of circles in the source image
		* @param img The source image of type \b CV_8UC1 or \b CV_8UC3.
		* @param threshold1 The higher threshold of the two, passed to the \a Canny edge detector (the lower one is twice smaller).
		* @param threshold2 The accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.
		* Circles, corresponding to the larger accumulator values, will be returned first.
		* @returns The number of circles in the source image
		*/
		DllExport size_t	getNumCircles(const Mat &img, double threshold1 = 100, double threshold2 = 30);
		/**
		* @brief
		*/
//		DllExport float	getTransparancy() { return 0; }
		/**
		* @brief
		*/
//		float getCompactness() { return 0; }
		/**
		* @brief
		*/
//		float getVariance() { return 0; }
		/**
		* @brief
		*/
//		int	getArea() { return 0; }
		/**
		* @brief
		*/
//		int	getPerimeter() { return 0; }
	}
} }