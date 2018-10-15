#include "Intensity.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CIntensity::get(const Mat &img, cv::Scalar weight)
{
	DGM_ASSERT_MSG(img.channels() == 3, "Input image has %d channel(s), but must have 3.", img.channels());

	// OpenCV function addWeighted() has a bug.
	Mat res(img.size(), CV_8UC1);
	for (int y = 0; y < img.rows; y++) {
		const byte  *pImg = img.ptr<byte>(y);
		byte		*pRes = res.ptr<byte>(y);
		for (int x = 0; x < img.cols; x++) {
			double sum = 0;
			for (int c = 0; c < 3; c++)
				sum += weight.val[c] * pImg[3 * x + c];
			pRes[x] = static_cast<byte> (MIN(255, MAX(0, sum + 0.5f)));
		} // x
	} // y

	return res;
}
} }