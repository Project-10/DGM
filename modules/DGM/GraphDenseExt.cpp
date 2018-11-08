#include "GraphDenseExt.h"
#include "GraphDense.h"
#include "macroses.h"
#include "densecrf/edgePotentialPotts.h"

namespace DirectGraphicalModels 
{
    void CGraphDenseExt::addNodes(Size graphSize)
    {
        if (m_graph.getNumNodes() != 0) m_graph.reset();
        m_size = graphSize;

        Mat pots(graphSize, CV_32FC1, Scalar(1.0f / m_graph.getNumStates()));
        
        m_graph.addNodes(pots.clone().reshape(1, pots.cols * pots.rows));
    }
    
    void CGraphDenseExt::setNodes(const Mat &pots)
	{
        m_size = pots.size();

        if (m_graph.getNumNodes() == pots.cols * pots.rows) m_graph.setNodes(pots.clone().reshape(1, pots.cols * pots.rows));
        else {
            if (m_graph.getNumNodes() != 0) m_graph.reset();
            m_graph.addNodes(pots.clone().reshape(1, pots.cols * pots.rows));
        }
	}

	void CGraphDenseExt::addGaussianEdgeModel(Vec2f s, float weight, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction)
	{
        Mat features(m_size.height * m_size.width, 2, CV_32FC1);
		int n = 0;
		for (int y = 0; y < m_size.height; y++)
			for (int x = 0; x < m_size.width; x++) {
				float *pFeature = features.ptr<float>(n++);
                pFeature[0] = x * s.val[0] / m_size.width;
                pFeature[1] = y * s.val[1] / m_size.height;
			} // x

		m_graph.addEdgeModel(new CEdgePotentialPotts(features, weight, SemiMetricFunction));
	}

	void CGraphDenseExt::addBilateralEdgeModel(const Mat &img, Vec2f s, Vec3f srgb, float weight, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction)
	{
        DGM_ASSERT_MSG(img.size() == m_size, "Resilution of the train image does not equal to the graph size");
        Mat features(img.rows * img.cols, 5, CV_32FC1);
		int n = 0;
		for (int y = 0; y < img.rows; y++) {
			const byte *pImg = img.ptr<byte>(y);
			for (int x = 0; x < img.cols; x++) {
				float *pFeature = features.ptr<float>(n++);
				pFeature[0] = x * s.val[0] / m_size.width;
				pFeature[1] = y * s.val[1] / m_size.height;
				// TODO: feature vector may have much more channels
                pFeature[2] = pImg[x * 3 + 0] * srgb.val[0] / 255;
				pFeature[3] = pImg[x * 3 + 1] * srgb.val[1] / 255;
				pFeature[4] = pImg[x * 3 + 2] * srgb.val[2] / 255;
			} // x
		} // y
		m_graph.addEdgeModel(new CEdgePotentialPotts(features, weight, SemiMetricFunction));
	}
}
