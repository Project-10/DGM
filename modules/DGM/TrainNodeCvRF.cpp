#include "TrainNodeCvRF.h"
#include "RForest.h"
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
CTrainNodeCvRF::CTrainNodeCvRF(byte nStates, word nFeatures, int maxSamples) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)
{
	TrainNodeCvRFParams params	= TRAIN_NODE_CV_RF_PARAMS_DEFAULT;
	params.maxSamples			= maxSamples;
	init(params);
}

void CTrainNodeCvRF::init(TrainNodeCvRFParams params)
{
	m_vSamplesAcc		= vec_mat_t(m_nStates);
	m_vNumInputSamples	= vec_int_t(m_nStates, 0);
	m_pRF				= std::auto_ptr<CRForest>(new CRForest(m_nStates));
	//m_pPriors = new float[m_nStates];
	//std::fill(m_pPriors, m_pPriors + m_nStates, 1.0f);
	m_pParams		= std::auto_ptr<CvRTParams>(new CvRTParams(params.max_depth,
		params.min_sample_count,
		params.regression_accuracy,
		params.use_surrogates,
		params.max_categories,
		NULL, //m_pPriors,
		params.calc_var_importance,
		params.nactive_vars,
		params.maxCount,
		params.epsilon,
		params.term_criteria_type
		));

	if (params.maxSamples == 0) m_maxSamples = std::numeric_limits<int>::max();
	else						m_maxSamples = params.maxSamples;	
}

// Destructor
CTrainNodeCvRF::~CTrainNodeCvRF(void)
{
}

void CTrainNodeCvRF::reset(void)
{
	for (Mat &acc : m_vSamplesAcc) acc.release();
	std::fill(m_vNumInputSamples.begin(), m_vNumInputSamples.end(), 0);
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
	m_pRF->load(fileName.c_str());
}

void CTrainNodeCvRF::addFeatureVec(const Mat &featureVector, byte gt)
{
	// Assertions:
	DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %d is out of range %d", gt, m_nStates);

	if (m_vSamplesAcc[gt].rows < m_maxSamples) {
		m_vSamplesAcc[gt].push_back(featureVector.t());
	}
	else {
		int k = random::u(0, m_vNumInputSamples[gt]);
		if (k < m_maxSamples)
			m_vSamplesAcc[gt].row(k) = featureVector.t();
	}
	m_vNumInputSamples[gt]++;
}

void CTrainNodeCvRF::train(bool doClean)
{
#ifdef DEBUG_PRINT_INFO
	printf("\n");
#endif
	// Filling the <samples> and <classes>
	Mat samples, classes;
	for (byte s = 0; s < m_nStates; s++) {						// states
		int nSamples = m_vSamplesAcc[s].rows;
		DGM_ASSERT(nSamples <= m_maxSamples);

#ifdef DEBUG_PRINT_INFO		
		printf("State[%d] - %d of %d samples\n", s, nSamples, m_vNumInputSamples[s]);
#endif

		m_vSamplesAcc[s].convertTo(m_vSamplesAcc[s], CV_32FC1);

		samples.push_back(m_vSamplesAcc[s]);
		classes.push_back(Mat(nSamples, 1, CV_32FC1, Scalar(s)));
		if (doClean) m_vSamplesAcc[s].release();				// free memory
	} // s

    // Filling <var_type>
	Mat var_type(m_nFeatures + 1, 1, CV_8UC1, Scalar(CV_VAR_NUMERICAL));		// all inputs are numerical

	var_type.at<byte>(m_nFeatures, 0) = CV_VAR_CATEGORICAL;

	// Training
	try {
		m_pRF->train(samples, CV_ROW_SAMPLE, classes, Mat(), Mat(), var_type, Mat(), *m_pParams);
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
	transpose(fv, fv);

	potential.release();
	potential = m_pRF->predict(fv);
}
}