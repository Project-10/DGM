#include "GraphDenseExt.h"
#include "GraphDense.h"
#include "densecrf/edgePotential.h"

namespace DirectGraphicalModels 
{
	void CGraphDenseExt::setNodes(const Mat &pots)
	{
		if (m_graph.getNumNodes()) m_graph.reset();
		m_graph.addNodes(pots.clone().reshape(1, pots.cols * pots.rows));
	}

	void CGraphDenseExt::setEdgesGaussian(CvSize imgSize, float sx, float sy, float w , const SemiMetricFunction *pFunction)
	{
		Mat features(imgSize.height * imgSize.width, 2, CV_32FC1);
		int n = 0;
		for (int y = 0; y < imgSize.height; y++)
			for (int x = 0; x < imgSize.width; x++) {
				float *pFeature = features.ptr<float>(n++);
				pFeature[0] = x / sx;
				pFeature[1] = y / sy;
			} // x

		m_graph.setEdgeModel(new CEdgePotentialPotts(features, w, pFunction));
	}

	void CGraphDenseExt::setEdgesBilateral(const Mat &img, float sx, float sy, float sr, float sg, float sb, float w, const SemiMetricFunction *pFunction)
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
		m_graph.setEdgeModel(new CEdgePotentialPotts(features, w, pFunction));
	}
}