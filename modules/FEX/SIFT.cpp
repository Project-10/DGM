#include "SIFT.h"
#include "opencv\SIFT.h"

namespace DirectGraphicalModels { namespace fex
{
	Mat	CSIFT::get(const Mat &img)
	{
		// Converting to one channel image
		Mat	I;
		if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
		else img.copyTo(I);

		Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
		std::vector<KeyPoint> keyPoints;
		sift->detect(img, keyPoints);
		drawKeypoints(
			  img
			, keyPoints
			, img
//			, DrawMatchesFlags::DRAW_RICH_KEYPOINTS
		);
		Mat descriptors;
		sift->detectAndCompute(I, Mat(), keyPoints, descriptors, true);


		return img;
	}
} }