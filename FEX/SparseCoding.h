// Sparse Coding feature extraction class interface
// Written by Sergey G. Kosov in 2016 for Project X 
#pragma once

#include "BaseFeatureExtractor.h"
#include "SparseDictionary.h"
#include "SquareNeighborhood.h"

namespace DirectGraphicalModels { namespace fex 
{
	// ================================ SC Class ==============================
	/**
	* @brief Sparse Coding feature extraction class.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CSparseCoding : public CBaseFeatureExtractor, public CSparseDictionary
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC1.
		*/
		DllExport CSparseCoding(const Mat &img) : CBaseFeatureExtractor(img) {}
		DllExport virtual ~CSparseCoding(void) {}

		DllExport virtual Mat	get(void) const { return get(m_img, m_D); }

		/**
		* @brief Extracts the sparse coding feature.
		* @details For each pixel of the source image this function calculates the variance within the pixel's neighbourhood \a nbhd.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		* @param dictionary Dictionary \f$\mathbb{D}\f$:  Mat(size blockSize^2 x nWords; type CV_64FC1)
		* @param nbhd Neighborhood around the pixel, where the samples are estimated. (Ref. @ref SqNeighbourhood).
		* @return The sparse coding feature image of type \b CV_8UC{nWords}.
		*/
		DllExport static Mat	get(const Mat &img, const Mat &dictionary, SqNeighbourhood nbhd = sqNeighbourhood(3));
	};
} }
