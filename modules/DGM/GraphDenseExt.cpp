#include "GraphDenseExt.h"
#include "GraphDense.h"
#include "EdgeModelPotts.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
    void CGraphDenseExt::buildGraph(Size graphSize)
    {
        m_size = graphSize;
		
		if (m_graph.getNumNodes()) m_graph.reset();
        
		// 2D default potentials
		Mat pots(graphSize, CV_32FC1, Scalar(1.0f / m_graph.getNumStates()));
        m_graph.addNodes(pots.clone().reshape(1, pots.cols * pots.rows));
    }
    
    void CGraphDenseExt::setGraph(const Mat &pots)
	{
        m_size = pots.size();

        if (m_graph.getNumNodes() == pots.cols * pots.rows) 
			m_graph.setNodes(0, pots.clone().reshape(1, pots.cols * pots.rows));
        else {
            if (m_graph.getNumNodes()) m_graph.reset();
            m_graph.addNodes(pots.clone().reshape(1, pots.cols * pots.rows));
        }
	}

	void CGraphDenseExt::addGaussianEdgeModel(Vec2f sigma, float weight, const std::function<void(const Mat& src, Mat& dst)> &semiMetricFunction)
	{
		Mat features;
		Mat feature(1, 2, CV_32FC1);
		float *pFeature = feature.ptr<float>(0);
		for (int y = 0; y < m_size.height; y++) 
			for (int x = 0; x < m_size.width; x++) {
				pFeature[0] = x / sigma.val[0];
				pFeature[1] = y / sigma.val[1];
				features.push_back(feature);
			} // x

		m_graph.addEdgeModel(std::make_shared<CEdgeModelPotts>(features, weight, semiMetricFunction));
	}

	void CGraphDenseExt::addBilateralEdgeModel(const Mat &featureVectors, Vec2f sigma, float sigma_opt, float weight, const std::function<void(const Mat& src, Mat& dst)> &semiMetricFunction)
	{
        const word	nFeatures = featureVectors.channels();
        
        DGM_ASSERT_MSG(featureVectors.size() == m_size, "Resilution of the train image does not equal to the graph size");
        Mat features; 
        Mat feature(1, 2 + nFeatures, CV_32FC1);
        float *pFeature = feature.ptr<float>(0);
        for (int y = 0; y < m_size.height; y++) {
			const byte *pFv = featureVectors.ptr<byte>(y);
			for (int x = 0; x < m_size.width; x++) {
				pFeature[0] = x / sigma.val[0];
				pFeature[1] = y / sigma.val[1];
                for (word f = 0; f < nFeatures; f++)
                    pFeature[2 + f] = pFv[nFeatures * x + f] / sigma_opt;
                features.push_back(feature);
			} // x
		} // y
		m_graph.addEdgeModel(std::make_shared<CEdgeModelPotts>(features, weight, semiMetricFunction));
	}

    void CGraphDenseExt::addBilateralEdgeModel(const vec_mat_t &featureVectors, Vec2f sigma, float sigma_opt, float weight, const std::function<void(const Mat& src, Mat& dst)> &semiMetricFunction)
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
                pFeature[0] = x / sigma.val[0];
                pFeature[1] = y / sigma.val[1];
                for (int f = 0; f < nFeatures; f++)
                    pFeature[2 + f] = pFv[f][x] / sigma_opt;
                features.push_back(feature);
            } // x
        } // y
        m_graph.addEdgeModel(std::make_shared<CEdgeModelPotts>(features, weight, semiMetricFunction));
    }
}
