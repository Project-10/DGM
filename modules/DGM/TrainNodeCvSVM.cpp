#include "TrainNodeCvSVM.h"
#include "SamplesAccumulator.h"

namespace DirectGraphicalModels
{
	// Constructor
	CTrainNodeCvSVM::CTrainNodeCvSVM(byte nStates, word nFeatures, TrainNodeCvSVMParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
	{
		init(params);
	}

	// Constructor
	CTrainNodeCvSVM::CTrainNodeCvSVM(byte nStates, word nFeatures, size_t maxSamples) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
	{
		TrainNodeCvSVMParams params = TRAIN_NODE_CV_SVM_PARAMS_DEFAULT;
		params.maxSamples = maxSamples;
		init(params);
	}

	void CTrainNodeCvSVM::init(TrainNodeCvSVMParams params)
	{
		m_pSamplesAcc = new CSamplesAccumulator(m_nStates, params.maxSamples);

		m_pSVM = ml::SVM::create();
		// TODO: Set other parameters
		// m_pSVM->set...
	}

	// Destructor
	CTrainNodeCvSVM::~CTrainNodeCvSVM(void)
	{
		delete m_pSamplesAcc;
	}

	void	CTrainNodeCvSVM::reset(void)
	{
		m_pSamplesAcc->reset();
		m_pSVM->clear();
	}

	void	CTrainNodeCvSVM::save(const std::string &path, const std::string &name, short idx) const
	{
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvSVM" : name, idx);
		m_pSVM->save(fileName.c_str());
	}

	void	CTrainNodeCvSVM::load(const std::string &path, const std::string &name, short idx)
	{
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvSVM" : name, idx);
		m_pSVM = Algorithm::load<ml::SVM>(fileName.c_str());
	}

	void	CTrainNodeCvSVM::addFeatureVec(const Mat &featureVector, byte gt)
	{
		m_pSamplesAcc->addSample(featureVector, gt);
	}

	void	CTrainNodeCvSVM::train(bool doClean)
	{
	}

	void	CTrainNodeCvSVM::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
	}
}