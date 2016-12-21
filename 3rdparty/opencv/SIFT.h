/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "types.h"

namespace cv
{
	namespace xfeatures2d
	{
		/******************************* Defs and macros *****************************/
		static const int	SIFT_DESCR_WIDTH		= 4;						// default width of descriptor histogram array
		static const int	SIFT_DESCR_HIST_BINS	= 8;						// default number of bins per histogram in descriptor array
		static const float	SIFT_INIT_SIGMA			= 0.5f;						// assumed gaussian blur for input image
		static const int	SIFT_IMG_BORDER			= 5;						// width of border in which to ignore keypoints
		static const int	SIFT_MAX_INTERP_STEPS	= 5;						// maximum steps of keypoint interpolation before failure
		static const int	SIFT_ORI_HIST_BINS		= 36;						// default number of bins in histogram for orientation assignment
		static const float	SIFT_ORI_SIG_FCTR		= 1.5f;						// determines gaussian sigma for orientation assignment
		static const float	SIFT_ORI_RADIUS			= 3 * SIFT_ORI_SIG_FCTR;	// determines the radius of the region used in orientation assignment
		static const float	SIFT_ORI_PEAK_RATIO		= 0.8f;						// orientation magnitude relative to max that results in new feature
		static const float	SIFT_DESCR_SCL_FCTR		= 3.0f;						// determines the size of a single descriptor orientation histogram
		static const float	SIFT_DESCR_MAG_THR		= 0.2f;						// threshold on magnitude of elements of descriptor vector
		static const float	SIFT_INT_DESCR_FCTR		= 512.0f;					// factor used to convert floating-point descriptor to unsigned char

#if 0	// intermediate type used for DoG pyramids
		typedef short sift_wt;
		static const int	SIFT_FIXPT_SCALE		= 48;
#else	// intermediate type used for DoG pyramids
		typedef float sift_wt;
		static const int	SIFT_FIXPT_SCALE		= 1;
#endif

		//! @addtogroup xfeatures2d_nonfree
		//! @{

		/** @brief Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform
		(SIFT) algorithm by D. Lowe @cite Lowe04 .
		*/
		class CV_EXPORTS_W SIFT : public Feature2D
		{
		public:
			explicit SIFT(int _nfeatures = 0, int _nOctaveLayers = 3, double _contrastThreshold = 0.04, double _edgeThreshold = 10, double _sigma = 1.6)
				: nfeatures(_nfeatures)
				, nOctaveLayers(_nOctaveLayers)
				, contrastThreshold(_contrastThreshold)
				, edgeThreshold(_edgeThreshold)
				, sigma(_sigma)
			{}
			
			/**
			@param nfeatures The number of best features to retain. The features are ranked by their scores
			(measured in SIFT algorithm as the local contrast)

			@param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The
			number of octaves is computed automatically from the image resolution.

			@param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform
			(low-contrast) regions. The larger the threshold, the less features are produced by the detector.

			@param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning
			is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are
			filtered out (more features are retained).

			@param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image
			is captured with a weak camera with soft lenses, you might want to reduce the number.
			*/
			CV_WRAP static Ptr<SIFT> create(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04, double edgeThreshold = 10, double sigma = 1.6)
			{
				return makePtr<SIFT>(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
			}

			//! returns the descriptor size in floats (128)
			int descriptorSize(void) const { return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS; }

			//! returns the descriptor type
			int descriptorType(void) const { return CV_32F; }

			//! returns the default norm type
			int defaultNorm(void) const { return NORM_L2; }

			//! finds the keypoints and computes descriptors for them using SIFT algorithm.
			//! Optionally it can compute descriptors for the user-provided keypoints
			void detectAndCompute(InputArray img, InputArray mask, std::vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints = false);

			void buildGaussianPyramid(const Mat &base, std::vector<Mat> &pyr, int nOctaves) const;
			void buildDoGPyramid(const std::vector<Mat> &pyr, std::vector<Mat> &dogpyr) const;
			void findScaleSpaceExtrema(const std::vector<Mat> &gauss_pyr, const std::vector<Mat> &dog_pyr, std::vector<KeyPoint> &keypoints) const;


		protected:
			CV_PROP_RW int nfeatures;
			CV_PROP_RW int nOctaveLayers;
			CV_PROP_RW double contrastThreshold;
			CV_PROP_RW double edgeThreshold;
			CV_PROP_RW double sigma;
		};

		typedef SIFT SiftFeatureDetector;
		typedef SIFT SiftDescriptorExtractor;
	}
}