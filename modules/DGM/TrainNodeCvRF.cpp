#include "TrainNodeCvRF.h"
#include "RForest.h"
#include "SamplesAccumulator.h"
#include "random.h"
#include "macroses.h"
#include <limits>

namespace DirectGraphicalModels
{
// Constructor
CTrainNodeCvRF::CTrainNodeCvRF(byte nStates, word nFeatures, TrainNodeCvRFParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
{
	init(params);
}

// Constructor
CTrainNodeCvRF::CTrainNodeCvRF(byte nStates, word nFeatures, size_t maxSamples) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
{
	TrainNodeCvRFParams params	= TRAIN_NODE_CV_RF_PARAMS_DEFAULT;
	params.maxSamples			= maxSamples;
	init(params);
}

void CTrainNodeCvRF::init(TrainNodeCvRFParams params)
{
	m_pSamplesAcc = new CSamplesAccumulator(m_nStates, params.maxSamples);

	m_pRF				= CRForest::create();
	m_pRF->setMaxDepth(params.max_depth);
	m_pRF->setMinSampleCount(params.min_sample_count);
	m_pRF->setRegressionAccuracy(params.regression_accuracy);
	m_pRF->setUseSurrogates(params.use_surrogates);
	m_pRF->setMaxCategories(params.max_categories);
	m_pRF->setCalculateVarImportance(params.calc_var_importance);
	m_pRF->setActiveVarCount(params.nactive_vars);
	m_pRF->setTermCriteria(TermCriteria(params.term_criteria_type, params.maxCount, params.epsilon));

	m_pRF->setNumStates(m_nStates);
}

// Destructor
CTrainNodeCvRF::~CTrainNodeCvRF(void)
{ 
	delete m_pSamplesAcc;
}

void CTrainNodeCvRF::reset(void)
{
	m_pSamplesAcc->reset();
	m_pRF->clear();
}

void CTrainNodeCvRF::save(const std::string &path, const std::string &name, short idx) const
{
	std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvRF" : name, idx);
	m_pRF->save(fileName.c_str());
}

void CTrainNodeCvRF::load(const std::string &path, const std::string &name, short idx)
{
	std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvRF" : name, idx);
	m_pRF = Algorithm::load<CRForest>(fileName.c_str());
}

void CTrainNodeCvRF::addFeatureVec(const Mat &featureVector, byte gt)
{ 
	m_pSamplesAcc->addSample(featureVector, gt); 
}

void CTrainNodeCvRF::train(bool doClean)
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
		m_pRF->train(ml::TrainData::create(samples, ml::ROW_SAMPLE, classes, noArray(), noArray(), noArray(), var_type));
	} catch (std::exception &e) {
		printf("EXCEPTION: %s\n", e.what());
		printf("Try to reduce the maximal depth of the forest or switch to x64.\n");
		getchar();
		exit(-1);
	}
}

Mat	CTrainNodeCvRF::getFeatureImportance(void) const
{
	return m_pRF->getVarImportance();
}

void CTrainNodeCvRF::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
{
	Mat fv;
	featureVector.convertTo(fv, CV_32FC1);
	potential = m_pRF->predict(fv.t());
}

}