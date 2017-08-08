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
		m_pSamplesAcc	= new CSamplesAccumulator(m_nStates, params.maxSamples);
		m_pKNN			= ml::KNearest::create();
		// using ml::KNearest::KDTREE causes an OpenCV exception
		// this is committed as bug #8917
		// https://github.com/opencv/opencv/issues/8917
		m_pKNN->setAlgorithmType(ml::KNearest::BRUTE_FORCE);
		m_params		= params;
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
			m_pKNN->train(ml::TrainData::create(samples, ml::ROW_SAMPLE, classes, noArray(), noArray(), noArray(), var_type));
		}
		catch (std::exception &e) {
			printf("EXCEPTION: %s\n", e.what());
			getchar();
			exit(-1);
		}
	}
	
	void	CTrainNodeCvKNN::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
		Mat fv;
		featureVector.convertTo(fv, CV_32FC1);
		Mat result, neighborResponses;
		m_pKNN->findNearest(fv.t(), m_params.maxNeighbors, result, neighborResponses);
		
		float *pResponse = neighborResponses.ptr<float>(0);
		int n = neighborResponses.cols;
		for (int i = 0; i < n; i++) {
			byte s = static_cast<byte>(pResponse[i]);
			potential.at<float>(s, 0) += 1.0f;
		}
		if (n) potential /= n;
		potential += m_params.bias;
	}
}
