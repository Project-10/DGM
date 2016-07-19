#include "TrainNodeCvGMM.h"
#include "SamplesAccumulator.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constatnts
const double CTrainNodeCvGMM::MIN_COEFFICIENT_BASE = 32.0;
	
// Constructor
CTrainNodeCvGMM::CTrainNodeCvGMM(byte nStates, word nFeatures, TrainNodeCvGMMParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates), m_minCoefficient(1)
{
	init(params);
}

// Constructor
CTrainNodeCvGMM::CTrainNodeCvGMM(byte nStates, word nFeatures, byte numGausses) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates), m_minCoefficient(1)
{
	TrainNodeCvGMMParams params = TRAIN_NODE_CV_GMM_PARAMS_DEFAULT;
	params.numGausses = numGausses;
	init(params);
}

void CTrainNodeCvGMM::init(TrainNodeCvGMMParams params)
{
	m_pSamplesAcc = new CSamplesAccumulator[m_nStates];
	for (byte s = 0; s < m_nStates; s++) {
		Ptr<ml::EM> pEM = ml::EM::create();
		pEM->setClustersNumber(params.numGausses);
		pEM->setCovarianceMatrixType(params.covariance_matrix_type);
		pEM->setTermCriteria(TermCriteria(params.term_criteria_type, params.maxCount, params.epsilon));
		m_vpEM.push_back(pEM);
	}
}

CTrainNodeCvGMM::~CTrainNodeCvGMM(void)
{
	delete [] m_pSamplesAcc;
	for(byte s = 0; s < m_nStates; s++) delete m_vpEM[s];
	m_vpEM.clear();
}

void CTrainNodeCvGMM::reset(void)
{
	for (byte s = 0; s < m_nStates; s++) {
		m_pSamplesAcc[s].reset();
		m_vpEM[s]->clear();
	}
}

void CTrainNodeCvGMM::save(const std::string &path, const std::string &name, short idx) const
{
	for (byte s = 0; s < m_nStates; s++) {
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvGMM_" + std::to_string(s) : name + "_" + std::to_string(s), idx);
		m_vpEM[s]->save(fileName.c_str());
	}
}

void CTrainNodeCvGMM::load(const std::string &path, const std::string &name, short idx)
{
	for (byte s = 0; s < m_nStates; s++) {
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvGMM_" + std::to_string(s) : name + "_" + std::to_string(s), idx);
		try {
			m_vpEM[s] = Algorithm::load<ml::EM>(fileName.c_str());
		} catch (Exception &) { 
			printf("In file: %s\n", fileName.c_str());
		}
	}

	m_minCoefficient = std::pow(MIN_COEFFICIENT_BASE, m_nFeatures);
}

void CTrainNodeCvGMM::addFeatureVec(const Mat &featureVector, byte gt)
{
	// Assertions:
	DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %d is out of range %d", gt, m_nStates);
	
	Mat point(m_nFeatures, 1, CV_64FC1);
	featureVector.convertTo(point, point.type());
	m_pSamplesAcc[gt].addSample(point);
}

void CTrainNodeCvGMM::train(void)
{
	printf("\n");
	for (byte s = 0; s < m_nStates; s++) {								// states
		size_t nSamples = m_pSamplesAcc[s].getNumSamples();
		printf("State[%d] - %zu samples\n", s, nSamples);
		if (nSamples == 0) continue;
		Mat samples(static_cast<int>(nSamples), m_nFeatures, CV_64FC1);
		for (register unsigned int smp = 0; smp < nSamples; smp++) {	// samples
			double * pSamples = samples.ptr<double>(smp);
			Mat Sample = m_pSamplesAcc[s].getSample(smp);
			for (word f = 0; f < m_nFeatures; f++)				// features
				pSamples[f] = Sample.at<double>(f, 0);	
		} // smp
		if (!m_vpEM[s]->trainEM(samples)) printf("Error EM training!\n");
	} // s
	
	m_minCoefficient = std::pow(MIN_COEFFICIENT_BASE, m_nFeatures);
}

void CTrainNodeCvGMM::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
{
	Mat fv;
	featureVector.convertTo(fv, CV_64FC1);
	transpose(fv, fv);


	// Min Coefficient approach
	for (byte s = 0; s < m_nStates; s++) { 					// state
		float	* pPot	= potential.ptr<float>(s);
		byte	* pMask	= mask.ptr<byte>(s);		
		if (m_vpEM[s]->isTrained()) 
			pPot[0] = static_cast<float>(std::exp(m_vpEM[s]->predict2(fv, noArray())[0]) * m_minCoefficient);
		else {
			// pPot[0] = 0; 
			pMask[0] = 0;
		}
	} // s
		

	// Minimax approach
	/*double min = 1.0e+150;
	double max = 1.0e-150;
	double *v = new double[m_nStates];
	for (byte s = 0; s < m_nStates; s++) 	{		// state
		if (m_pEM[s].isTrained()) {
			v[s] = std::exp(m_pEM[s].predict(fv)[0]);
			if (max < v[s]) max = v[s];
			if (min > v[s]) min = v[s];
		}
	}
	for (byte s = 0; s < m_nStates; s++) {
		v[s] /= (max - min);
		res.at<float>(s, 0) = static_cast<float>(v[s]);
	}
	delete [] v;*/
}
}
