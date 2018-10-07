#include "GraphDenseExt.h"
#include "GraphDense.h"
#include "densecrf/edgepotential.h"

namespace DirectGraphicalModels 
{
	void CGraphDenseExt::addNodes(const Mat &pots)
	{
		if (m_graph.getNumNodes()) m_graph.reset();
		m_graph.addNodes(pots.clone().reshape(1, pots.cols * pots.rows));
	}

	void CGraphDenseExt::addGaussianEdgeModel(CvSize graphSize, float sx, float sy, float w, const std::function<void(const vec_float_t &src, vec_float_t &dst)> &SemiMetricFunction)
	{
		Mat features(graphSize.height * graphSize.width, 2, CV_32FC1);
		int n = 0;
		for (int y = 0; y < graphSize.height; y++)
			for (int x = 0; x < graphSize.width; x++) {
				float *pFeature = features.ptr<float>(n++);
				pFeature[0] = x / sx;
				pFeature[1] = y / sy;
			} // x

		m_graph.addEdgeModel(new CEdgePotentialPotts(features, w, SemiMetricFunction));
	}

	void CGraphDenseExt::addBilateralEdgeModel(const Mat &img, float sx, float sy, float sr, float sg, float sb, float w, const std::function<void(const vec_float_t &src, vec_float_t &dst)> &SemiMetricFunction)
	{
		Mat features(img.rows * img.cols, 5, CV_32FC1);
		int n = 0;
		for (int y = 0; y < img.rows; y++) {
			const byte *pImg = img.ptr<byte>(y);
			for (int x = 0; x < img.cols; x++) {
				float *pFeature = features.ptr<float>(n++);
				pFeature[0] = x / sx;
				pFeature[1] = y / sy;
				pFeature[2] = pImg[x * 3 + 0] / sr;
				pFeature[3] = pImg[x * 3 + 1] / sg;
				pFeature[4] = pImg[x * 3 + 2] / sb;
			} // x
		} // y
		m_graph.addEdgeModel(new CEdgePotentialPotts(features, w, SemiMetricFunction));
	}
}
