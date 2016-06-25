// Sparse Coding feature extraction class interface
// Written by Sergey G. Kosov in 2016 for Project X (based on Xingdi (Eric) Yuan implementation) http://eric-yuan.me/sc/
#pragma once

#include "BaseFeatureExtractor.h"
#include "SparseCodeDictionary.h"
#include "SquareNeighborhood.h"

namespace DirectGraphicalModels { namespace fex 
{
	class CSparseCodeDictionary;

	// ================================ SC Class ==============================
	/**
	* @brief Sparse Coding feature extraction class. (http://www.scholarpedia.org/article/Sparse_coding)
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CSparseCode : public CBaseFeatureExtractor, private CSparseCodeDictionary
	{
	public:
		/**
		* @brief Constructor.
		* @param img Input image of type \b CV_8UC1.
		*/
		DllExport CSparseCode(const Mat &img) : CBaseFeatureExtractor(img) {}
		DllExport virtual ~CSparseCode(void) {}

		DllExport virtual Mat	get(void) const { return get(m_img, nullptr); }

		/**
		* @brief Extracts the sparse coding feature.
		* @details For each pixel of the source image this function calculates the variance within the pixel's neighbourhood \a nbhd.
		* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
		* @param pDict Pointer to the dictionary
		* @param nbhd Neighborhood around the pixel, where the variance is estimated. (Ref. @ref SqNeighbourhood).
		* @return The sparse coding feature image of type \b CV_8UC{nWords}.
		*/
		DllExport static Mat	get(const Mat &img, CSparseCodeDictionary *pDict, SqNeighbourhood nbhd = sqNeighbourhood(3));
	};
} }
