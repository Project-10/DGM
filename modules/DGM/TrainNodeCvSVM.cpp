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
		m_pSVM->setType(ml::SVM::C_SVC);
		m_pSVM->setKernel(ml::SVM::RBF);
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
			classes.push_back(Mat(nSamples, 1, CV_32FC1, Scalar(s)));
			if (doClean) m_pSamplesAcc->release(s);				// free memory
		} // s
		samples.convertTo(samples, CV_32FC1);

		// Filling <var_type>
		Mat var_type(m_nFeatures + 1, 1, CV_8UC1, Scalar(ml::VAR_NUMERICAL));		// all inputs are numerical
		var_type.at<byte>(m_nFeatures, 0) = ml::VAR_CATEGORICAL;

		// Training
		try {
			m_pSVM->trainAuto(ml::TrainData::create(samples, ml::ROW_SAMPLE, classes, noArray(), noArray(), noArray(), var_type));
		}
		catch (std::exception &e) {
			printf("EXCEPTION: %s\n", e.what());
			getchar();
			exit(-1);
		}
	}

	void	CTrainNodeCvSVM::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
		Mat fv;
		featureVector.convertTo(fv, CV_32FC1);
		byte s = static_cast<byte>(m_pSVM->predict(fv.t()));
		potential.at<float>(s, 0) = 1.0f;
		potential += 0.1f;
	}
}