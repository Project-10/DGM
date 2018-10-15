#include "HSV.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CHSV::get(const Mat &img)
{
	DGM_ASSERT_MSG(img.channels() == 3, "Input image has %d channel(s), but must have 3.", img.channels());
	Mat res;
	cvtColor(img, res, cv::ColorConversionCodes::COLOR_BGR2HSV);
	return res;
}
} }