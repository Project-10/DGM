#include "TrainNodeKNN.h"
#include "KDTree.h"
#include "SamplesAccumulator.h"
#include "mathop.h"

namespace DirectGraphicalModels 
{
	// Constructor
	CTrainNodeKNN::CTrainNodeKNN(byte nStates, word nFeatures, TrainNodeKNNParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
	{
		init(params);
	}

	// Constructor
	CTrainNodeKNN::CTrainNodeKNN(byte nStates, word nFeatures, size_t maxSamples) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
	{
		TrainNodeKNNParams params = TRAIN_NODE_KNN_PARAMS_DEFAULT;
		params.maxSamples = maxSamples;
		init(params);
	}
	
	void CTrainNodeKNN::init(TrainNodeKNNParams params)
	{
		m_pSamplesAcc = new CSamplesAccumulator(m_nStates, params.maxSamples);
		m_pTree = new CKDTree();
		m_params = params;
	}

	// Destructor
	CTrainNodeKNN::~CTrainNodeKNN(void)
	{
		delete m_pSamplesAcc;
		delete m_pTree;
	}

	void CTrainNodeKNN::reset(void)
	{
		m_pSamplesAcc->reset();
		m_pTree->reset();
	}

	void CTrainNodeKNN::save(const std::string &path, const std::string &name, short idx) const
	{
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeKNN" : name, idx);
		m_pTree->save(fileName);
	}

	void CTrainNodeKNN::load(const std::string &path, const std::string &name, short idx)
	{
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeKNN" : name, idx);
		m_pTree->load(fileName);
	}

	void CTrainNodeKNN::addFeatureVec(const Mat &featureVector, byte gt)
	{
		m_pSamplesAcc->addSample(featureVector, gt);
	}

	void CTrainNodeKNN::train(bool doClean)
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


	/// @todo Generate a 2D histogram for this method
	void CTrainNodeKNN::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
		auto nearestNeighbors = m_pTree->findNearestNeighbors(featureVector.t(), m_params.maxNeighbors);
		potential.setTo(m_params.bias);
		float minr = mathop::Euclidian<byte, float>(featureVector.t(), nearestNeighbors.front()->getKey());

		size_t n = nearestNeighbors.size();
		for (auto node : nearestNeighbors) {
			float r = mathop::Euclidian<byte, float>(featureVector.t(), node->getKey());
			byte  s = node->getValue();				
			r = 1 + r - minr;
			potential.at<float>(s, 0) += 1.0f / (r * r);
			//potential.at<float>(s, 0) += 1.0f / n;
		}
	}
}