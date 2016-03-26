// HOG feature extraction class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "BaseFeatureExtractor.h"
#include "SquareNeighborhood.h"

namespace DirectGraphicalModels { namespace fex
{
	// ================================ HOG Class ==============================
	/**
	* @brief HOG (<a href="http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients">histogram of oriented gradients</a>) feature extraction class.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/		
	class CHOG : public CBaseFeatureExtractor
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		*/
		DllExport CHOG(const Mat &img) : CBaseFeatureExtractor(img) {}
		DllExport virtual ~CHOG(void) {}

		DllExport virtual Mat	get(void) const {return get(m_img);}

		/**
		* @brief Extracts the HOG feature.
		* @details For each pixel of the source image this function calculates the histogram of oriented gradients inside the pixel's neighbourhood \a nbhd.
		* The histogram consists of \a nBins values, it is normalized, and stored as \a nBins channel image, thus, the channel index corresponds to the histogram index.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		* @param nBins Number of bins. Hence a single bin covers an angle of \f$\frac{180^\circ}{nBins}\f$.
		* @param nbhd Neighborhood around the pixel, where its histogram is estimated. (Ref. @ref SqNeighbourhood).
		* @return The HOG feature image of type \b CV_8UC{n}, where \f$n=nBins\f$.
		*/
		DllExport static Mat	get(const Mat &img, int nBins = 9, SqNeighbourhood nbhd = sqNeighbourhood(5));
	};
} }