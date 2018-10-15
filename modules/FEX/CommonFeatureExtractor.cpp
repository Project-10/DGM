#include "CommonFeatureExtractor.h"

namespace DirectGraphicalModels { namespace fex
{
CCommonFeatureExtractor CCommonFeatureExtractor::invert(void) const
{
	Mat res;
	bitwise_not(m_img, res);
	return CCommonFeatureExtractor(res);
}

CCommonFeatureExtractor CCommonFeatureExtractor::blur(int R) const
{
	Mat res;
	R = 2 * R + 1;
	GaussianBlur(m_img, res, cv::Size(R, R), 0.0, 0.0, BORDER_REFLECT);
	return CCommonFeatureExtractor(res);
}

CCommonFeatureExtractor CCommonFeatureExtractor::autoContrast(void) const
{
	DGM_ASSERT_MSG(m_img.depth() == CV_8U, "The source image must have 8-bit / channel depth");
	Mat res;
	vec_mat_t vChannels;
	split(m_img, vChannels);
#ifdef ENABLE_PPL
	concurrency::parallel_for_each(vChannels.begin(), vChannels.end(), [](Mat &c) {
#else
	for (Mat &c : vChannels) {
#endif
		double minVal, maxVal;
		minMaxLoc(c, &minVal, &maxVal);
		double k = (maxVal > minVal) ? k = 255.0 / (maxVal - minVal) : 1.0;
		c.convertTo(c, c.type(), k, -minVal * k);
	}
#ifdef ENABLE_PPL
	);
#endif
	merge(vChannels, res);
	return CCommonFeatureExtractor(res);
}

CCommonFeatureExtractor CCommonFeatureExtractor::thresholding(byte threshold) const
{
	// Converting to one channel image
	Mat res;
	if (m_img.channels() != 1) cvtColor(m_img, res, cv::ColorConversionCodes::COLOR_RGB2GRAY);
	else m_img.copyTo(res);

	for (int y = 0; y < res.rows; y++) {
		byte *pRes = res.ptr<byte>(y);
		for (int x = 0; x < res.cols; x++) 
			pRes[x] = (pRes[x] > threshold) ? 225 : 0;
	} // y

	return CCommonFeatureExtractor(res);
}

CCommonFeatureExtractor CCommonFeatureExtractor::getChannel(int channel) const
{
	DGM_ASSERT_MSG(channel < m_img.channels(), "The required channel %d does not exist in the %d-channel source image", channel, m_img.channels());
	Mat res;
	vec_mat_t vChannels;
	split(m_img, vChannels);
	vChannels.at(channel).copyTo(res);
	vChannels.clear();
	return CCommonFeatureExtractor(res);
}
} }