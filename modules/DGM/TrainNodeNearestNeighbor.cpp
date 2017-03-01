#include "TrainNodeNearestNeighbor.h"
#include "SamplesAccumulator.h"
#include "mathop.h"

namespace DirectGraphicalModels 
{
	// Constructor
	CTrainNodeNearestNeighbor::CTrainNodeNearestNeighbor(byte nStates, word nFeatures) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates) 
	{
		m_pSamplesAcc = new CSamplesAccumulatorPairs(nStates);
	}
	
	// Destructor
	CTrainNodeNearestNeighbor::~CTrainNodeNearestNeighbor(void) 
	{
		delete m_pSamplesAcc;
	}

	void CTrainNodeNearestNeighbor::reset(void) 
	{
		m_pSamplesAcc->reset();
	}

	void CTrainNodeNearestNeighbor::addFeatureVec(const Mat &featureVector, byte gt) 
	{
		m_pSamplesAcc->addSample(featureVector, gt);
	}

	void CTrainNodeNearestNeighbor::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const 
	{
		vec_samplePair_t vSamplePairs = m_pSamplesAcc->getSamplesContainer();
		//std::vector<std::pair<float, byte>> vDist;

		float D = -1;
		byte sol = 0;
		for (auto &pair : vSamplePairs) {
			float dist = mathop::Euclidian<byte, float>(featureVector, pair.first);
			if (D < 0) D = dist;
			else if (dist < D) {
				D = dist;
				sol = pair.second;
			}
			//vDist.push_back(std::make_pair(dist, pair.second));
		}

		//std::sort(vDist.begin(), vDist.end(), [](auto &left, auto &right) { return left.first > right.first; });

		//for (int i = 0; i < 100; i++) {
		//	byte state = vDist[i].second;
		//	potential.at<float>(state, 0) += 1.0f / 100;
		//}

		potential.setTo(10);
		potential.at<float>(sol, 0) = 100;
	}
}