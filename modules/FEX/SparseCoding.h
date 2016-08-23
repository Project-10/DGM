// Sparse Coding feature extraction class interface
// Written by Sergey G. Kosov in 2016 for Project X 
#pragma once

#include "BaseFeatureExtractor.h"
#include "SparseDictionary.h"
#include "SquareNeighborhood.h"

namespace DirectGraphicalModels {
	namespace fex
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

			DllExport virtual Mat		get(void) const { return get(m_img, getDictionary()); }

			/**
			* @brief Extracts the sparse coding feature.
			* @details > This fuction supports dictionaries with \a nWords less or equal to 512 words. For larger dictionaries use CSparseCoding::get_v() function directly.
			* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
			* @param D Sparse dictionary \f$D\f$:  Mat(size nWords x blockSize^2; type CV_32FC1).
			* > Dictionary should be learned from a training data with CSparseDictionary::train() function,<br>
			* > or it may be loaded directed from a \a dic file with CSparseDictionary::getDictionary("dictionary.dic").
			* @param nbhd Neighborhood around the pixel, where the samples are estimated. (Ref. @ref SqNeighbourhood). It shoul be a square with a side equal to blockSize.
			* @return The sparse coding feature image of type \b CV_8UC{nWords}.
			*/
			DllExport static Mat		get(const Mat &img, const Mat &D, SqNeighbourhood nbhd = sqNeighbourhood(3));
			/**
			* @brief Extracts the sparse coding feature.
			* @details This function is an alternative to get(), which can handle large amount of features (more then 512)
			* @param img Input image of type \b CV_8UC1 or \b CV_8UC3.
			* @param D Sparse dictionary \f$D\f$:  Mat(size nWords x blockSize^2; type CV_32FC1).
			* > Dictionary should be learned from a training data with CSparseDictionary::train() function,<br>
			* > or it may be loaded directed from a \a dic file with CSparseDictionary::getDictionary("dictionary.dic").
			* @param nbhd Neighborhood around the pixel, where the samples are estimated. (Ref. @ref SqNeighbourhood). It shoul be a square with a side equal to blockSize.
			* @return The vector with \a nWords sparse coding feature images of type \b CV_8UC1 each.
			*/
			DllExport static vec_mat_t	get_v(const Mat &img, const Mat &D, SqNeighbourhood nbhd = sqNeighbourhood(3));
		};
	}
}
