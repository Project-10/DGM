#include "Distance.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CDistance::get(const Mat &img, byte thres, double multiplier)
{
	Mat I, res;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);		// Converting to one channel image
	else img.copyTo(I);
	
	threshold(I, I, thres, 255, THRESH_BINARY_INV);
	distanceTransform(I, I, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	I.convertTo(res, CV_8UC1, multiplier);
	return res;
}
} }
