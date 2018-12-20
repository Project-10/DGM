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

        if (m_graph.getNumNodes() == pots.cols * pots.rows) m_graph.setNodes(0, pots.clone().reshape(1, pots.cols * pots.rows));
        else {
            if (m_graph.getNumNodes() != 0) m_graph.reset();
            m_graph.addNodes(pots.clone().reshape(1, pots.cols * pots.rows));
        }
	}

	void CGraphDenseExt::addGaussianEdgeModel(Vec2f sigma, float weight, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction)
	{
        Mat features(m_size.height * m_size.width, 2, CV_32FC1);
		int n = 0;
		for (int y = 0; y < m_size.height; y++)
			for (int x = 0; x < m_size.width; x++) {
				float *pFeature = features.ptr<float>(n++);
                pFeature[0] = x / sigma.val[0];
				pFeature[1] = y / sigma.val[1];
			} // x

		m_graph.addEdgeModel(new CEdgePotentialPotts(features, weight, SemiMetricFunction));
	}

	void CGraphDenseExt::addBilateralEdgeModel(const Mat &featureVectors, Vec2f s, float srgb, float weight, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction)
	{
        const word	nFeatures = featureVectors.channels();
        
        DGM_ASSERT_MSG(featureVectors.size() == m_size, "Resilution of the train image does not equal to the graph size");
        Mat features; 
        Mat feature(1, 2 + nFeatures, CV_32FC1);
        float *pFeature = feature.ptr<float>(0);
        for (int y = 0; y < m_size.height; y++) {
			const byte *pFv = featureVectors.ptr<byte>(y);
			for (int x = 0; x < m_size.width; x++) {
				pFeature[0] = x * s.val[0] / m_size.width;
				pFeature[1] = y * s.val[1] / m_size.height;
                for (word f = 0; f < nFeatures; f++)
                    pFeature[2 + f] = pFv[nFeatures * x + f] * srgb / 255;
                features.push_back(feature);
			} // x
		} // y
		m_graph.addEdgeModel(new CEdgePotentialPotts(features, weight, SemiMetricFunction));
	}

    void CGraphDenseExt::addBilateralEdgeModel(const vec_mat_t &featureVectors, Vec2f s, float srgb, float weight, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction)
    {
        const word	nFeatures = static_cast<word>(featureVectors.size());
        
        DGM_ASSERT_MSG(!featureVectors.empty(), "The train image is empty");
        DGM_ASSERT_MSG(featureVectors[0].size() == m_size, "Resilution of the train image does not equal to the graph size");
        Mat features;
        Mat feature(1, 2 + nFeatures, CV_32FC1);
        float *pFeature = feature.ptr<float>(0);
        for (int y = 0; y < m_size.height; y++) {
            byte const **pFv = new const byte *[nFeatures];
            for (word f = 0; f < nFeatures; f++) pFv[f] = featureVectors[f].ptr<byte>(y);
            for (int x = 0; x < m_size.width; x++) {
                pFeature[0] = x * s.val[0] / m_size.width;
                pFeature[1] = y * s.val[1] / m_size.height;
                for (int f = 0; f < nFeatures; f++)
                    pFeature[2 + f] = pFv[f][x] * srgb / 255;
                features.push_back(feature);
            } // x
        } // y
        m_graph.addEdgeModel(new CEdgePotentialPotts(features, weight, SemiMetricFunction));
    }
}
