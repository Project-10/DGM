#include "GraphPairwiseExt.h"
#include "GraphPairwise.h"
#include "TrainEdge.h"
#include "TrainEdgePottsCS.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
	void CGraphPairwiseExt::addDefaultEdgesModel(float val, float weight)
	{
        const byte	nStates = m_pGraphML->getGraph().getNumStates();
		const Mat	pot	= CTrainEdge::getDefaultEdgePotentials(sqrtf(val), nStates);
		m_pGraphML->getGraph().setEdges({}, pot);
	}

	// TODO:
	void CGraphPairwiseExt::addDefaultEdgesModel(const Mat &featureVectors, float val, float weight)
	{
        const byte nStates = m_pGraphML->getGraph().getNumStates();
        const word nFeatures = featureVectors.channels();
        const CTrainEdge &edgeTrainer = CTrainEdgePottsCS(nStates, nFeatures);
        fillEdges(edgeTrainer, featureVectors, { val, 0.01f }, weight);
	}

	// TODO:
    void CGraphPairwiseExt::addDefaultEdgesModel(const vec_mat_t &featureVectors, float val, float weight)
    {
        const byte nStates = m_pGraphML->getGraph().getNumStates();
        const word nFeatures = static_cast<word>(featureVectors.size());
        const CTrainEdge &edgeTrainer = CTrainEdgePottsCS(nStates, nFeatures);
        fillEdges(edgeTrainer, featureVectors, { val, 0.01f }, weight);
    }
}