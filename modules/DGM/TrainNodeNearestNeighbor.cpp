#include "TrainNodeNearestNeighbor.h"
#include "SamplesAccumulator.h"

namespace DirectGraphicalModels 
{
	// Constructor
	CTrainNodeNearestNeighbor::CTrainNodeNearestNeighbor(byte nStates, word nFeatures) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates) 
	{
		m_pSamplesAcc = new CSamplesAccumulator(nStates);
	}
	
	// Destructor
	CTrainNodeNearestNeighbor::~CTrainNodeNearestNeighbor(void) 
	{
		delete m_pSamplesAcc;
	}

	void CTrainNodeNearestNeighbor::reset(void) {
		m_pSamplesAcc->reset();
	}

	void CTrainNodeNearestNeighbor::addFeatureVec(const Mat &featureVector, byte gt) {
		m_pSamplesAcc->addSample(featureVector, gt);
	}
}