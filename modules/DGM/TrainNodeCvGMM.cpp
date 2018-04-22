#include "TrainNodeCvGMM.h"
#include "SamplesAccumulator.h"
#include "random.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constatnts
const double CTrainNodeCvGMM::MIN_COEFFICIENT_BASE = 32.0;
	
// Constructor
CTrainNodeCvGMM::CTrainNodeCvGMM(byte nStates, word nFeatures, TrainNodeCvGMMParams params) :  CBaseRandomModel(nStates), CTrainNode(nStates, nFeatures), m_minCoefficient(1)
{
	init(params);
}

// Constructor
CTrainNodeCvGMM::CTrainNodeCvGMM(byte nStates, word nFeatures, size_t maxSamples, byte numGausses) : CBaseRandomModel(nStates), CTrainNode(nStates, nFeatures), m_minCoefficient(1)
{
	TrainNodeCvGMMParams params = TRAIN_NODE_CV_GMM_PARAMS_DEFAULT;
	params.maxSamples = maxSamples;
	params.numGausses = numGausses;
	init(params);
}

void CTrainNodeCvGMM::init(TrainNodeCvGMMParams params)
{
	m_pSamplesAcc = new CSamplesAccumulator(m_nStates, params.maxSamples);

	for (byte s = 0; s < m_nStates; s++) {
		Ptr<ml::EM> pEM = ml::EM::create();
		pEM->setClustersNumber(params.numGausses);
		pEM->setCovarianceMatrixType(params.covariance_matrix_type);
		pEM->setTermCriteria(TermCriteria(params.term_criteria_type, params.maxCount, params.epsilon));
		m_vpEM.push_back(pEM);
	}
}

// Destructor
CTrainNodeCvGMM::~CTrainNodeCvGMM(void)
{
	delete m_pSamplesAcc;
	m_vpEM.clear();
}

void CTrainNodeCvGMM::reset(void)
{
	m_pSamplesAcc->reset();
	for (Ptr<ml::EM> &em : m_vpEM) em->clear();
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
	m_pSamplesAcc->addSample(featureVector, gt);
}

void CTrainNodeCvGMM::train(bool doClean)
{
#ifdef DEBUG_PRINT_INFO
	printf("\n");
#endif

	for (byte s = 0; s < m_nStates; s++) {						// states
		int nSamples = m_pSamplesAcc->getNumSamples(s);
#ifdef DEBUG_PRINT_INFO		
		printf("State[%d] - %d of %d samples\n", s, nSamples, m_pSamplesAcc->getNumInputSamples(s));
#endif
		if (nSamples == 0) continue;
		DGM_IF_WARNING(!m_vpEM[s]->trainEM(m_pSamplesAcc->getSamplesContainer(s)), "Error EM training!");
		if (doClean) m_pSamplesAcc->release(s);
	} // s
	
	m_minCoefficient = std::pow(MIN_COEFFICIENT_BASE, m_nFeatures);
}

void CTrainNodeCvGMM::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
{
	Mat fv;
	featureVector.convertTo(fv, CV_64FC1);

	// Min Coefficient approach
	for (byte s = 0; s < m_nStates; s++) { 					// state
		float	* pPot	= potential.ptr<float>(s);
		byte	* pMask	= mask.ptr<byte>(s);		
		if (m_vpEM[s]->isTrained()) 
			pPot[0] = static_cast<float>(std::exp(m_vpEM[s]->predict2(fv.t(), noArray())[0]) * m_minCoefficient);
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
