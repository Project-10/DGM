#include "GraphPairwiseExt.h"

namespace DirectGraphicalModels 
{
	// Constructor	
	CGraphPairwiseExt::CGraphPairwiseExt(CGraphPairwise &graph, byte gType) 
		: CGraphExt()
		, m_pGraphML(new CGraphLayered(graph, 1, gType)) 
	{}

	void CGraphPairwiseExt::addNodes(Size graphSize)
    {
        m_pGraphML->addNodes(graphSize);
    }

	void CGraphPairwiseExt::setNodes(const Mat &pots)
	{
		m_pGraphML->setNodes(pots, Mat());
	}

	void CGraphPairwiseExt::addDefaultEdgesModel(float val, float weight = 1.0f)
	{
//        const byte nStates = m_pGraphML->getGraph().getNumStates();
//
//        // Assertions
//		DGM_ASSERT(m_pGraphML->getSize().width * m_pGraphML->getSize().height == m_pGraphML->getGraph().getNumNodes());
//
//		Mat ePot = CTrainEdge::getDefaultEdgePotentials(val, nStates);
//#ifdef ENABLE_PPL
//        concurrency::parallel_for(0, m_pGraphML->getSize().height, [&](int y) {
//#else 
//        for (int y = 0; y < m_pGraphML->getSize().height; y++) {
//#endif
//            for (int x = 0; x < m_pGraphML->getSize().width; x++) {
//                size_t idx = y * m_pGraphML->getSize().width + x;
//                if (m_pGraphML->getType() & GRAPH_EDGES_GRID) {
//                    if (x > 0)												m_pGraphML->getGraph().setArc(idx, idx - 1, ePot);
//                    if (y > 0)												m_pGraphML->getGraph().setArc(idx, idx - 1 * m_pGraphML->getSize().width, ePot);
//                } // edges_grid
//
//                if (m_pGraphML->getType() & GRAPH_EDGES_DIAG) {
//                    if ((x > 0) && (y > 0))									m_pGraphML->getGraph().setArc(idx, idx - m_pGraphML->getSize().width - 1, ePot);
//                    if ((x < m_pGraphML->getSize().width - 1) && (y > 0))	m_pGraphML->getGraph().setArc(idx, idx - m_pGraphML->getSize().width + 1, ePot);
//                } // edges_diag
//            } // x
//#ifdef ENABLE_PPL
//        }); // y
//#else
//        } // y
//#endif	
	}

	void CGraphPairwiseExt::addDefaultEdgesModel(const Mat &featureVectors, float val, float weight)
	{
        //const byte nStates = m_pGraphML->getGraph().getNumStates();
        //const word nFeatures = featureVectors.channels();
        //const CTrainEdge &edgeTrainer = CTrainEdgePottsCS(nStates, nFeatures);
        //fillEdges(&edgeTrainer, featureVectors, { val, 0.01f }, weight);
	}

    void CGraphPairwiseExt::addDefaultEdgesModel(const vec_mat_t &featureVectors, float val, float weight)
    {
        //const byte nStates = m_pGraphML->getGraph().getNumStates();
        //const word nFeatures = static_cast<word>(featureVectors.size());
        //const CTrainEdge &edgeTrainer = CTrainEdgePottsCS(nStates, nFeatures);
        //fillEdges(&edgeTrainer, featureVectors, { val, 0.01f }, weight);
    }

	Size CGraphPairwiseExt::getSize(void) const 
	{
        return m_pGraphML->getSize();
    }

}