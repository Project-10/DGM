#include "TrainNodeCvKNN.h"
#include "SamplesAccumulator.h"

namespace DirectGraphicalModels
{
	// Constructor
	CTrainNodeCvKNN::CTrainNodeCvKNN(byte nStates, word nFeatures, TrainNodeCvKNNParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
	{
		init(params);
	}

	// Constructor
	CTrainNodeCvKNN::CTrainNodeCvKNN(byte nStates, word nFeatures, size_t maxSamples) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
	{
		TrainNodeCvKNNParams params = TRAIN_NODE_CV_KNN_PARAMS_DEFAULT;
		params.maxSamples = maxSamples;
		init(params);
	}

	void CTrainNodeCvKNN::init(TrainNodeCvKNNParams params)
	{
		m_pSamplesAcc = new CSamplesAccumulator(m_nStates, params.maxSamples);

		m_pKNN = ml::KNearest::create();
		// TODO: set other params
		//m_pKNN->set...
	}

	// Destructor
	CTrainNodeCvKNN::~CTrainNodeCvKNN(void)
	{
		delete m_pSamplesAcc;
	}
	
	void	CTrainNodeCvKNN::reset(void)
	{
		m_pSamplesAcc->reset();
		m_pKNN->clear();
	}
	
	void	CTrainNodeCvKNN::save(const std::string &path, const std::string &name, short idx) const
	{
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvKNN" : name, idx);
		m_pKNN->save(fileName.c_str());
	}
	
	void	CTrainNodeCvKNN::load(const std::string &path, const std::string &name, short idx)
	{
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvKNN" : name, idx);
		m_pKNN = Algorithm::load<ml::KNearest>(fileName.c_str());
	}
	
	void	CTrainNodeCvKNN::addFeatureVec(const Mat &featureVector, byte gt)
	{
		m_pSamplesAcc->addSample(featureVector, gt);
	}
	
	void	CTrainNodeCvKNN::train(bool doClean)
	{
	}
	
	void	CTrainNodeCvKNN::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
	}
}
