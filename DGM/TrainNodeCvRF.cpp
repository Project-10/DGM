#include "TrainNodeCvRF.h"
#include "SamplesAccumulator.h"
#include "RForest.h"
#include "Random.h"
#include "macroses.h"
#include <limits>

namespace DirectGraphicalModels
{
// Constructor
CTrainNodeCvRF::CTrainNodeCvRF(byte nStates, byte nFeatures, TrainNodeCvRFParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
{
	init(params);
}

// Constructor
CTrainNodeCvRF::CTrainNodeCvRF(byte nStates, byte nFeatures, size_t maxSamples) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
{
	TrainNodeCvRFParams params	= TRAIN_NODE_CV_RF_PARAMS_DEFAULT;
	params.maxSamples			= maxSamples;
	init(params);
}

void CTrainNodeCvRF::init(TrainNodeCvRFParams params)
{
	m_pSamplesAcc	= new CSamplesAccumulator[m_nStates];
	m_pRF			= CRForest::create();
	m_pRF->setMaxDepth(params.max_depth);
	m_pRF->setMinSampleCount(params.min_sample_count);
	m_pRF->setRegressionAccuracy(params.regression_accuracy);
	m_pRF->setUseSurrogates(params.use_surrogates);
	m_pRF->setMaxCategories(params.max_categories);
	m_pRF->setCalculateVarImportance(params.calc_var_importance);
	m_pRF->setActiveVarCount(params.nactive_vars);
	m_pRF->setTermCriteria(TermCriteria(params.term_criteria_type, params.maxCount, params.epsilon));

	m_pRF->setNumStates(m_nStates);

	if (params.maxSamples == 0) m_maxSamples = std::numeric_limits<size_t>::max();
	else						m_maxSamples = params.maxSamples;	
}

// Destructor
CTrainNodeCvRF::~CTrainNodeCvRF(void)
{
	delete [] m_pSamplesAcc;
	delete m_pRF;
}

void CTrainNodeCvRF::reset(void)
{
	for (byte s = 0; s < m_nStates; s++) m_pSamplesAcc[s].reset();
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
	// Assertions:
	DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %d is out of range %d", gt, m_nStates);
	
	if (m_pSamplesAcc[gt].getNumSamples() < m_maxSamples) {
		Mat point(m_nFeatures, 1, CV_64FC1);
		featureVector.convertTo(point, point.type());
		m_pSamplesAcc[gt].addSample(point);
	}
}

void CTrainNodeCvRF::train(void)
{
	register byte	s, f;									// state and feature indexes 
	register size_t	smp;									// sample index

	size_t nAllSamples = 0;
	for (s = 0; s < m_nStates; s++) {						// states
		size_t nSamples = m_pSamplesAcc[s].getNumSamples();
		size_t S	    =  MIN(nSamples, m_maxSamples);
		nAllSamples += S;
	} // i
	
	CRandom random;

	Mat samples(static_cast<int>(nAllSamples), m_nFeatures, CV_32FC1);
	Mat classes(static_cast<int>(nAllSamples), 1          , CV_32FC1);	
	
	// printf("\n");
	int l = 0;
	for (s = 0; s < m_nStates; s++) {						// states
		size_t nSamples = m_pSamplesAcc[s].getNumSamples();
		size_t S		= MIN(nSamples, m_maxSamples);
		
		//printf("State[%d] - %d samples\n", i, nSamples);

		for (smp = 0; smp < S; smp++, l++) {				// samples
			float *pSamples = samples.ptr<float>(l);
			size_t sample = (nSamples > m_maxSamples) ? random.du() % nSamples : smp;
			Mat Sample = m_pSamplesAcc[s].getSample(sample);
			for (f = 0; f < m_nFeatures; f++) 				// features
				pSamples[f] = static_cast<float>(Sample.at<double>(f, 0));
			classes.at<float>(l, 0) = s;	
		} // smp
	} // s

	Mat var_type = Mat(m_nFeatures + 1, 1, CV_8U);
	var_type.setTo(Scalar(ml::VAR_NUMERICAL)); // all inputs are numerical
	var_type.at<uchar>(m_nFeatures, 0) = ml::VAR_CATEGORICAL;

	m_pRF->train(ml::TrainData::create(samples, ml::ROW_SAMPLE, classes, noArray(), noArray(), noArray(), var_type));
}

Mat	CTrainNodeCvRF::getFeatureImportance(void) const
{
	return m_pRF->getVarImportance();
}

void CTrainNodeCvRF::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
{
	Mat fv;
	featureVector.convertTo(fv, CV_32FC1);
	transpose(fv, fv);

	potential.release();
	potential = m_pRF->predict(fv);

}

}