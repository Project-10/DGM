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
		* @return The number of staight lines in the source image
		*/
		DllExport size_t	getNumLines(const Mat &img, int threshold1 = 100, int threshold2 = 50);
		/**
		* @brief Returns the number of circles in the source image.
		* @param img The source image of type \b CV_8UC1 or \b CV_8UC3.
		* @param threshold1 The higher threshold of the two, passed to the \a Canny edge detector (the lower one is twice smaller).
		* @param threshold2 The accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.
		* Circles, corresponding to the larger accumulator values, will be returned first.
		* @return The number of circles in the source image
		*/
		DllExport size_t	getNumCircles(const Mat &img, int threshold1 = 100, int threshold2 = 30);
		/**
		* @brief Returns the weighted-mean transparancy of the source image
		* @details The weighted-mean transparancy is evaluated as follows:
		* \f[ \frac{1}{width \times hight} \sum^{width}_{x = 1}\sum^{height}_{y = 1} (1 - d_{x,y}) \cdot (img_{x,y} - \mu), \f]
		* where \f$d_{x,y}\f$ is the normalized distance between a position \f$(x,y)\f$ and the image center.
		* @param img The source image of type \b CV_8UC1 or \b CV_8UC3.
		* @return The weighted-mean transparancy of the source image.
		*/
		DllExport float		getOpacity(const Mat &img);
		/**
		* @brief Retunrs the variance of the source image.
		* @param img The source image of type \b CV_8UC1 or \b CV_8UC3.
		* @return The variance of the source image.
		*/
		DllExport float		getVariance(const Mat &img);
		/**
		* @brief Returns the number of non-zero pixels in the source image.
		* @param img The source image of type \b CV_8UC1 or \b CV_8UC3.
		* @return The number of non-zero pixels in the source image.
		*/
		DllExport int		getArea(const Mat &img);
		/**
		* @brief Returns the perimeter of an object in the source image.
		* @details This function retunrs the number of edge pixels, assuming a pixel to belong to an edge if any of its neighboring pixels have different value.
		* @param img The source image of type \b CV_8UC1 or \b CV_8UC3.
		* @return The number of edge pixels in the source image.
		*/
		DllExport int		getPerimeter(const Mat &img);
		/**
		* @brief Returns the compactness of the object in the source image.
		* @details The compactness is calculates as follows: \f$\frac{P^2}{4\Pi S}\f$, where \a P and \a S are perimeter and area of the object, respectively.
		* @param img The source image of type \b CV_8UC1 or \b CV_8UC3.
		* @return The compactness of the object in the source image.
		*/
		DllExport float		getCompactness(const Mat &img);
	}
} }