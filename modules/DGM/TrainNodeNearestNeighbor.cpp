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
		m_pTree			= new CKDTree();
	}
	
	// Destructor
	CTrainNodeNearestNeighbor::~CTrainNodeNearestNeighbor(void) 
	{
		delete m_pSamplesAcc;
		delete m_pTree;
	}

	void CTrainNodeNearestNeighbor::reset(void) 
	{
		m_pSamplesAcc->reset();
		m_pTree->reset();
	}

	void CTrainNodeNearestNeighbor::addFeatureVec(const Mat &featureVector, byte gt) 
	{
		m_pSamplesAcc->addSample(featureVector, gt);
	}

	void CTrainNodeNearestNeighbor::train(bool doClean)
	{
#ifdef DEBUG_PRINT_INFO
		printf("\n");
#endif

		// Filling the <samples> and <classes>
		Mat samples, classes;
		for (byte s = 0; s < m_nStates; s++) {						// states
			int nSamples = m_pSamplesAcc->getNumSamples(s);
#ifdef DEBUG_PRINT_INFO		
			printf("State[%d] - %d of %d samples\n", s, nSamples, m_pSamplesAcc->getNumInputSamples(s));
#endif
			samples.push_back(m_pSamplesAcc->getSamplesContainer(s));
			classes.push_back(Mat(nSamples, 1, CV_8UC1, Scalar(s)));
			if (doClean) m_pSamplesAcc->release(s);				// free memory
		} // s

		// Training, e.g. building the tree
		m_pTree->build(samples, classes);
	}

	void CTrainNodeNearestNeighbor::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
		byte val = m_pTree->findNearestNeighbor(featureVector.t())->getValue();
		potential.setTo(10);
		potential.at<float>(val, 0) = 100;
	}
}