#include "GraphDense2D.h"

void CGraphDense2D::setNodes(const Mat &pots)
{
	CGraphDense::setNodes(reinterpret_cast<const float *>(pots.data), pots.cols * pots.rows);
}

void CGraphDense2D::setEdgesGaussian(CvSize imgSize, float sx, float sy, float w, const SemiMetricFunction *function)
{
	Mat feature(imgSize, CV_32FC2);
	for (int y = 0; y < feature.rows; y++) {
		float *pFeature = feature.ptr<float>(y);
		for (int x = 0; x <feature.cols; x++) {
			pFeature[x * 2 + 0] = x / sx;
			pFeature[x * 2 + 1] = y / sy;
		} // x
	}// y
	setEdgesPotts(reinterpret_cast<float *>(feature.data), 2, w, function);
}

void CGraphDense2D::setEdgesBilateral(const Mat &img, float sx, float sy, float sr, float sg, float sb, float w, const SemiMetricFunction * function)
{
	Mat feature(img.size(), CV_MAKE_TYPE(CV_32F, 5));
	for (int y = 0; y < feature.rows; y++) {
		const byte *pImg = img.ptr<byte>(y);
		float *pFeature = feature.ptr<float>(y);
		for (int x = 0; x < img.cols; x++) {
			pFeature[x * 5 + 0] = x / sx;
			pFeature[x * 5 + 1] = y / sy;
			pFeature[x * 5 + 2] = pImg[x * 3 + 0] / sr;
			pFeature[x * 5 + 3] = pImg[x * 3 + 1] / sg;
			pFeature[x * 5 + 4] = pImg[x * 3 + 2] / sb;
		} // x
	} // y

	setEdgesPotts(reinterpret_cast<float *>(feature.data), 5, w, function);
}
