#include "TrainNodeCvANN.h"
#include "SamplesAccumulator.h"

namespace DirectGraphicalModels
{
	// Constructor
	CTrainNodeCvANN::CTrainNodeCvANN(byte nStates, word nFeatures, TrainNodeCvANNParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
	{
		init(params);
	}

	// Constructor
	CTrainNodeCvANN::CTrainNodeCvANN(byte nStates, word nFeatures, size_t maxSamples) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
	{
		TrainNodeCvANNParams params = TRAIN_NODE_CV_ANN_PARAMS_DEFAULT;
		params.maxSamples = maxSamples;
		init(params);
	}

	void CTrainNodeCvANN::init(TrainNodeCvANNParams params)
	{
		m_pSamplesAcc = new CSamplesAccumulator(m_nStates, params.maxSamples);
		m_pANN = ml::ANN_MLP::create();
		// TODO: Set other parameters
		m_pANN->setLayerSizes(std::vector<int>({ m_nFeatures, 16 * m_nStates, 8 * m_nStates, 4 * m_nStates, m_nStates}));
		m_pANN->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0.0, 0.0);
		m_pANN->setTermCriteria(TermCriteria(params.term_criteria_type, params.maxCount, params.epsilon));
		m_pANN->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
	}

	// Destructor
	CTrainNodeCvANN::~CTrainNodeCvANN(void)
	{
		delete m_pSamplesAcc;
	}

	void	CTrainNodeCvANN::reset(void)
	{
		m_pSamplesAcc->reset();
		m_pANN->clear();
	}

	void	CTrainNodeCvANN::save(const std::string &path, const std::string &name, short idx) const
	{
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvANN" : name, idx);
		m_pANN->save(fileName.c_str());
	}

	void	CTrainNodeCvANN::load(const std::string &path, const std::string &name, short idx)
	{
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvANN" : name, idx);
		m_pANN = Algorithm::load<ml::ANN_MLP>(fileName.c_str());
	}

	void	CTrainNodeCvANN::addFeatureVec(const Mat &featureVector, byte gt)
	{
		m_pSamplesAcc->addSample(featureVector, gt);
	}

	void	CTrainNodeCvANN::train(bool doClean)
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
			Mat classes_s(nSamples, m_nStates, CV_32FC1, Scalar(0.0f));
			classes_s.col(s).setTo(1.0f);
			classes.push_back(classes_s);
			if (doClean) m_pSamplesAcc->release(s);				// free memory
		} // s
		samples.convertTo(samples, CV_32FC1);

		// Training
		try {
			m_pANN->train(samples, ml::ROW_SAMPLE, classes);
		}
		catch (std::exception &e) {
			printf("EXCEPTION: %s\n", e.what());
			getchar();
			exit(-1);
		}
	}

	void	CTrainNodeCvANN::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
		Mat fv;
		featureVector.convertTo(fv, CV_32FC1);
		Mat test;
		byte s = static_cast<byte>(m_pANN->predict(fv.t(), test));
		//printf("test: %d x %d\n", test.cols, test.rows);
		//for (int i = 0; i < test.cols; i++) printf("%.3f ", test.at<float>(0, i));
		//printf("\n");

		test += 1;
		test = test.t();
		test.copyTo(potential);

		//potential.at<float>(s, 0) = 1.0f;
		//potential += 0.1f;
	}
}