#include "TrainNodeNearestNeighbor.h"
#include "SamplesAccumulator.h"
#include "KDTree.h"
#include "mathop.h"

namespace DirectGraphicalModels 
{
	// Constructor
	CTrainNodeNearestNeighbor::CTrainNodeNearestNeighbor(byte nStates, word nFeatures, size_t maxSamples) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
	{
		m_pSamplesAcc	= new CSamplesAccumulator(nStates, maxSamples);
		//m_pKDTree		= new CKDTree<100>();
	}
	
	// Destructor
	CTrainNodeNearestNeighbor::~CTrainNodeNearestNeighbor(void) 
	{
		delete m_pSamplesAcc;
		//delete m_pKDTree;
	}

	void CTrainNodeNearestNeighbor::reset(void) 
	{
		m_pSamplesAcc->reset();
	}

	void CTrainNodeNearestNeighbor::addFeatureVec(const Mat &featureVector, byte gt) 
	{
		m_pSamplesAcc->addSample(featureVector, gt);
	}

	void CTrainNodeNearestNeighbor::train(bool doClean)
	{
		// Build k-D Tree here
	}

	void CTrainNodeNearestNeighbor::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const 
	{
		float minDist = -1.0f;
		byte minState;
		
		
		for (byte s = 0; s < m_nStates; s++) {				// states
			int nSamples = m_pSamplesAcc->getNumSamples(s);
			
			if (nSamples == 0) {
				mask.at<byte>(s, 0) = 0;
				continue;
			}
			
			for (int smp = 0; smp < nSamples; smp++) {		// samples
				Mat sample = m_pSamplesAcc->getSamplesContainer(s).row(smp).t();
				float dist = mathop::Euclidian<byte, float>(featureVector, sample);
				
				if (minDist < -0.5f) {
					minDist = dist;
					minState = s;
				}
				else if (minDist > dist) {
					minDist = dist;
					minState = s;
				}
			} // smp 

			potential.at<float>(s, 0) = 10.0f;
		} // s


		potential.at<float>(minState, 0) = 100;
	}
}