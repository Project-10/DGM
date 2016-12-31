#include "TrainNodeCvGMM.h"
#include "random.h"
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
CTrainNodeCvGMM::CTrainNodeCvGMM(byte nStates, word nFeatures, int maxSamples, byte numGausses) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates), m_minCoefficient(1)
{
	TrainNodeCvGMMParams params = TRAIN_NODE_CV_GMM_PARAMS_DEFAULT;
	params.maxSamples = maxSamples;
	params.numGausses = numGausses;
	init(params);
}

void CTrainNodeCvGMM::init(TrainNodeCvGMMParams params)
{
	m_vSamplesAcc = vec_mat_t(m_nStates);
	m_vNumInputSamples = vec_int_t(m_nStates, 0);
	
	m_vpEM.reserve(m_nStates);
	for (byte s = 0; s < m_nStates; s++) 
		m_vpEM.push_back(std::auto_ptr<EM>(new EM(params.numGausses, params.covariance_matrix_type, TermCriteria(params.term_criteria_type, params.maxCount, params.epsilon))));

	if (params.maxSamples == 0) m_maxSamples = std::numeric_limits<int>::max();
	else						m_maxSamples = params.maxSamples;
}

CTrainNodeCvGMM::~CTrainNodeCvGMM(void)
{
	m_vpEM.clear();
}

void CTrainNodeCvGMM::reset(void)
{
	for (Mat &acc : m_vSamplesAcc) acc.release();
	std::fill(m_vNumInputSamples.begin(), m_vNumInputSamples.end(), 0);
	for (auto &em : m_vpEM) em->clear();
}

void CTrainNodeCvGMM::save(const std::string &path, const std::string &name, short idx) const
{
	for (byte s = 0; s < m_nStates; s++) {
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvGMM_" + std::to_string(s) : name + "_" + std::to_string(s), idx);
		FileStorage fs(fileName.c_str(), FileStorage::WRITE);
		m_vpEM[s]->write(fs);
	}
}

void CTrainNodeCvGMM::load(const std::string &path, const std::string &name, short idx)
{
	for (byte s = 0; s < m_nStates; s++) {
		std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvGMM_" + std::to_string(s) : name + "_" + std::to_string(s), idx);
		FileStorage fs(fileName.c_str(), FileStorage::READ);
		try {
			m_vpEM[s]->read(fs["StatModel.EM"]);
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

void CTrainNodeCvGMM::train(bool doClean)
{
#ifdef DEBUG_PRINT_INFO
	printf("\n");
#endif

	for (byte s = 0; s < m_nStates; s++) {						// states
		int nSamples = m_vSamplesAcc[s].rows;
		DGM_ASSERT(nSamples <= m_maxSamples);
#ifdef DEBUG_PRINT_INFO		
		printf("State[%d] - %d of %d samples\n", s, nSamples, m_vNumInputSamples[s]);
#endif
		if (nSamples == 0) continue;
		DGM_IF_WARNING(!m_vpEM[s]->train(m_vSamplesAcc[s]), "Error EM training!");
		if (doClean) m_vSamplesAcc[s].release();
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
			pPot[0] = static_cast<float>(std::exp(m_vpEM[s]->predict(fv)[0]) * m_minCoefficient);
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
