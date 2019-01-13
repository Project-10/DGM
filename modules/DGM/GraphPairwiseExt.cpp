#include "GraphPairwiseExt.h"
#include "IGraphPairwise.h"
#include "TrainEdge.h"
#include "TrainEdgePottsCS.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
	void CGraphPairwiseExt::addDefaultEdgesModel(float val, float weight)
	{
        if (weight != 1.0f) val = powf(val, weight);
		const byte	nStates = getGraph().getNumStates();
		getGraph().setEdges({}, CTrainEdge::getDefaultEdgePotentials(sqrtf(val), nStates));
	}

	void CGraphPairwiseExt::addDefaultEdgesModel(const Mat &featureVectors, float val, float weight)
	{
        const byte nStates = getGraph().getNumStates();
        const word nFeatures = featureVectors.channels();
		const CTrainEdgePottsCS edgeTrainer(nStates, nFeatures);
        fillEdges(edgeTrainer, featureVectors, { val, 0.001f }, weight);
	}

    void CGraphPairwiseExt::addDefaultEdgesModel(const vec_mat_t &featureVectors, float val, float weight)
    {
        const byte nStates = getGraph().getNumStates();
        const word nFeatures = static_cast<word>(featureVectors.size());
        const CTrainEdgePottsCS edgeTrainer(nStates, nFeatures);
        fillEdges(edgeTrainer, featureVectors, { val, 0.001f }, weight);
    }
}