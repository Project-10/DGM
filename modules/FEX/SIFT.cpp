#include "SIFT.h"
#include "opencv\SIFT.h"
#include "LinearMapper.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
	Mat	CSIFT::get(const Mat &img)
	{
		register int	i;						// feature index
		register int	x, y;
		int				width	= img.cols;
		int				height	= img.rows;
		
		// Converting to one channel image
		Mat	I;
		if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
		else img.copyTo(I);

		Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
		
		// Prepare key points
		std::vector<KeyPoint> vKeyPoints;
//		sift->detect(img, vKeyPoints);
		for (y = 0; y < height; y++)
			for (x = 0; x < width; x++)
				vKeyPoints.push_back(KeyPoint(static_cast<float>(x), static_cast<float>(y), 1.0f));

//		drawKeypoints(img, vKeyPoints, img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		Mat descriptors;
		sift->detectAndCompute(I, Mat(), vKeyPoints, descriptors, true);

		const int nFeatures = descriptors.cols;			// 128
		DGM_ASSERT(nFeatures == 128);
		DGM_ASSERT(nFeatures < CV_CN_MAX);

		// Initializing features
		vec_mat_t vFeatures(nFeatures);
		for (i = 0; i < nFeatures; i++) vFeatures[i].create(img.size(), CV_8UC1);

		std::vector<byte *> pFeatures(nFeatures);
		
		for (y = 0; y < height; y++) {
			for (i = 0; i < nFeatures; i++) pFeatures[i] = vFeatures[i].ptr<byte>(y);
			for (x = 0; x < width; x++) {
				float *pDescriptors = descriptors.ptr<float>(y * width + x);
				for (i = 0; i < nFeatures; i++) 
					pFeatures[i][x] = linear_mapper<byte>(pDescriptors[i], 0.0f, 255.0f);			// features[i] (x, y) = descriptors (i, pixel_idx)
			} // x
		} // y

		Mat res;
		merge(vFeatures, res);

		return res;
	}
} }