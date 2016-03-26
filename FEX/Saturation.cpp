#include "Saturation.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CSaturation::get(const Mat &img)
{
	DGM_ASSERT_MSG(img.channels() == 3, "Input image has %d channel(s), but must have 3.", img.channels());
	Mat res, hsv;
	cvtColor(img, hsv, CV_RGB2HSV);
	vec_mat_t vChannels(3);
	split(hsv, vChannels);
	vChannels.at(1).copyTo(res);
	vChannels.clear();
	return res;
}
} }