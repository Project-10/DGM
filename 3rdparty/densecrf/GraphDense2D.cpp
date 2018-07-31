#include "GraphDense2D.h"

void CGraphDense2D::setNodes(const Mat &pots)
{
    const float *pData = reinterpret_cast<const float *>(pots.data);
    CGraphDense::setNodes(vec_float_t(pData, pData + pots.cols * pots.rows * pots.channels()));
}

void CGraphDense2D::setEdgesGaussian(CvSize imgSize, float sx, float sy, float w, const SemiMetricFunction *pFunction)
{
	Mat feature(imgSize, CV_32FC2);
	for (int y = 0; y < feature.rows; y++) {
		float *pFeature = feature.ptr<float>(y);
		for (int x = 0; x <feature.cols; x++) {
			pFeature[x * 2 + 0] = x / sx;
			pFeature[x * 2 + 1] = y / sy;
		} // x
	}// y
    setEdges(new CEdgePotentialPotts(reinterpret_cast<float *>(feature.data), 2, feature.rows * feature.cols, w, pFunction));
}

void CGraphDense2D::setEdgesBilateral(const Mat &img, float sx, float sy, float sr, float sg, float sb, float w, const SemiMetricFunction *pFunction)
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
	setEdges(new CEdgePotentialPotts(reinterpret_cast<float *>(feature.data), 5, feature.rows * feature.cols, w, pFunction));
}
