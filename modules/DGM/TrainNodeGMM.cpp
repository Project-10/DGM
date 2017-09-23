#include "TrainNodeGMM.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	// Constructor
	CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures) 
		: CTrainNode(nStates, nFeatures)
		, CBaseRandomModel(nStates)
	{
		m_vGaussianMixtures.resize(nStates);
	}

	// Constructor
	CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures, byte maxGausses) 
		: CTrainNode(nStates, nFeatures) 
		, CBaseRandomModel(nStates)
	{
		m_vGaussianMixtures.resize(nStates);
	}

	void CTrainNodeGMM::reset(void) {
		m_vGaussianMixtures.clear();
	}
	
	void CTrainNodeGMM::addFeatureVec(const Mat &featureVector, byte gt) {
		
		Mat point;
		featureVector.convertTo(point, CV_64FC1);

		if (m_vGaussianMixtures[gt].empty()) m_vGaussianMixtures[gt].emplace_back(point);
		else {

		}
	}

	void CTrainNodeGMM::train(bool doClean) {}

	void CTrainNodeGMM::saveFile(FILE *pFile) const {}
	
	void CTrainNodeGMM::loadFile(FILE *pFile) {}

	void CTrainNodeGMM::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const {}
}
	
	
	
	
//// Constants
//const size_t		CTrainNodeGMM::MIN_SAMPLES		= 16;
//const long double	CTrainNodeGMM::MAX_COEFFICIENT =  1.0;
//
//// Constructor
//CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures, TrainNodeGMMParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates), m_minCoefficient(1), m_params(params)
//{
//	//m_vGMs.resize(nStates);
//	//for (auto &vGausses : m_vGMs)
//	//	vGausses.reserve(m_params.maxGausses);
//	//if (m_params.min_samples < MIN_SAMPLES) m_params.min_samples = MIN_SAMPLES;
//}
//
//// Constructor
//CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures, byte maxGausses) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates), m_minCoefficient(1), m_params(TRAIN_NODE_GMM_PARAMS_DEFAULT)
//{
//	//m_params.maxGausses = maxGausses;
//	//m_vGMs.resize(nStates);
//	//for (auto &vGausses : m_vGMs)
//	//	vGausses.reserve(m_params.maxGausses);
//}
//
//void CTrainNodeGMM::reset(void)
//{
//	//m_vGMs.clear();
//	//m_minCoefficient = 1;
//}
//
//void CTrainNodeGMM::addFeatureVec(const Mat &featureVector, byte gt)
//{
//	//Point	minPoint;
//
//	//// Assertions:
//	//DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %u is out of range [0; %u)", gt, m_nStates);
//	//
//	//Mat point(m_nFeatures, 1, CV_64FC1);
//	//featureVector.convertTo(point, point.type());
//
//	//if (m_vGMs[gt].empty()) m_vGMs[gt].emplace_back(point);				// NEW GAUSS
//	//else {
//	//	Mat dist = getDistance(gt, point);								// Calculate distances all existing Gaussians in the mixture to the point
//
//	//	// Finding the smallest distance
//	//	double minDist;
//	//	minMaxLoc(dist, &minDist, NULL, &minPoint, NULL);	
//	//	int gaussIdx = minPoint.y;
//	//	dist.release();
//
//	//	double dist_treshold;
//	//	if (m_params.dist_Mtreshold < 0)	dist_treshold = m_params.dist_Etreshold;
//	//	else								dist_treshold = m_params.dist_Mtreshold;
//	//	
//	//	// Add to existing Gaussian or crete a new one
//	//	if ((minDist > dist_treshold) && (m_vGMs[gt].size() < m_params.maxGausses)) m_vGMs[gt].emplace_back(point);		// NEW GAUSS
//	//	else {
//	//		CKDGauss &gauss = m_vGMs[gt][gaussIdx];					// the nearest Gaussian
//	//		if (true)	// TODO: check it
//	//			gauss.addPoint(point);								// update the nearest Gauss
//	//		else
//	//			gauss += CKDGauss(point);							// update the nearest Gauss
//
//	//		// Chech the updated Gauss function if after update it became too close to another Gauss function
//	//		if ((m_params.div_KLtreshold > 0) && (gauss.getNumPoints() >= m_params.min_samples)) {
//	//			Mat div = getDivergence(gt, gauss);
//	//			div.at<double>(gaussIdx, 0) = DBL_MAX;					// divergence to itself
//	//			
//	//			// Finding the smallest divergence
//	//			double minDiv;
//	//			minMaxLoc(div, &minDiv, NULL, &minPoint, NULL);	
//	//			int minGaussIdx = minPoint.y;
//	//			div.release();	
//
//	//			// Merge together if they are too close
//	//			if ((minGaussIdx >= 0) && (minDiv < m_params.div_KLtreshold)) {
//	//				m_vGMs[gt].at(minGaussIdx) += gauss;
//	//				m_vGMs[gt].erase(m_vGMs[gt].begin() + gaussIdx);
//	//			}
//	//		}
//	//	}
//	//}
//	//
//	//point.release();
//}
//
//void CTrainNodeGMM::train(bool)
//{
////	register byte	s;
////
////	// merge gausses with too small number of samples 
////	for (s = 0; s < m_nStates; s++) {							// state
////		for (auto it = m_vGMs[s].begin(); it != m_vGMs[s].end(); it++) {
////			it->freeze();
////			size_t nPoints = it->getNumPoints();
////			if (nPoints < m_params.min_samples) {				// if Gaussian is not full
////				word g = static_cast<word>(it - m_vGMs[s].begin());
////				if (nPoints >= MIN_SAMPLES) {
////					Mat div = getDivergence(s, &(*it)); 
////					div.at<double>(g, 0) = DBL_MAX;				// distance to itself (redundant here)
////
////					// Finding the smallest divergence
////					double minDiv;
////					Point minPoint;
////					minMaxLoc(div, &minDiv, NULL, &minPoint, NULL);	
////					int gaussIdx = minPoint.y;
////					div.release();				
////					if (gaussIdx >= 0) m_vGMs[s].at(gaussIdx) += (*it);
////				} // if Gaussian has less then MIN_SAMPLES points, consider it as a noise and delete
////				m_vGMs[s].erase(it);
////				it--;
////			} // if Gaussian Full
////		} // gausses
////	} // s
////
////	// getting the coefficients
////	for (s = 0; s < m_nStates; s++) {				// state
////		for (auto itGauss = m_vGMs[s].begin(); itGauss != m_vGMs[s].end(); itGauss++) {
////			itGauss->freeze();
////			long double Coefficient = itGauss->getAlpha();
////			if (Coefficient > MAX_COEFFICIENT) {			// i.e. if (Coefficient = \infinitiy) delete Gaussian
////				m_vGMs[s].erase(itGauss);
////				itGauss--;
////				continue;
////			}
////			if (Coefficient < m_minCoefficient)
////				m_minCoefficient = Coefficient;
////		} // gausses
////	} // s
////
////#ifdef DEBUG_PRINT_INFO
////	showStatus();
////#endif	// DEBUG_PRINT_INFO
//}
//
//void CTrainNodeGMM::saveFile(FILE *pFile) const
//{
//	//// m_params
//	//fwrite(&m_params.maxGausses,	 sizeof(word),	 1, pFile);
//	//fwrite(&m_params.min_samples,	 sizeof(size_t), 1, pFile);
//	//fwrite(&m_params.dist_Etreshold, sizeof(double), 1, pFile);
//	//fwrite(&m_params.dist_Mtreshold, sizeof(double), 1, pFile);
//	//fwrite(&m_params.div_KLtreshold, sizeof(double), 1, pFile);
//
//	//// m_pvpGausses;							
//	//for (byte s = 0; s < m_nStates; s++) {				// state
//	//	word nGausses = static_cast<word>(m_vGMs[s].size());
//	//	fwrite(&nGausses, sizeof(word), 1, pFile);
//	//	for (CKDGauss &gauss : m_vGMs[s]) {
//	//		size_t	nPoints = gauss.getNumPoints();
//	//		Mat		mu		= gauss.getMu();
//	//		Mat		sigma	= gauss.getSigma();
//
//	//		fwrite(&nPoints, sizeof(long), 1, pFile);
//	//		for (word y = 0; y < m_nFeatures; y++)
//	//			fwrite(&mu.at<double>(y, 0), sizeof(double), 1, pFile);
//	//		for (word y = 0; y < m_nFeatures; y++)
//	//			for (word x = 0; x < m_nFeatures; x++)
//	//				fwrite(&sigma.at<double>(y, x), sizeof(double), 1, pFile);
//	//		mu.release();
//	//		sigma.release();
//	//	} // gauss
//	//} // s
//
//	//fwrite(&m_minCoefficient, sizeof(long double), 1, pFile);
//}
//
//void CTrainNodeGMM::loadFile(FILE *pFile)
//{
//	//// m_params
//	//fread(&m_params.maxGausses,		sizeof(word),	1, pFile);
//	//fread(&m_params.min_samples,	sizeof(size_t), 1, pFile);
//	//fread(&m_params.dist_Etreshold, sizeof(double), 1, pFile);
//	//fread(&m_params.dist_Mtreshold, sizeof(double), 1, pFile);
//	//fread(&m_params.div_KLtreshold,	sizeof(double), 1, pFile);
//
//	//// m_pvpGausses;
//	//for (byte s = 0; s < m_nStates; s++) {				// state
//	//	word nGausses;
//	//	fread(&nGausses, sizeof(word), 1, pFile);
//	//	m_vGMs[s].assign(nGausses, CNDGauss(m_nFeatures));
//	//	for (CNDGauss &gauss : m_vGMs[s]) {
//	//		long nPoints;
//	//		Mat mu(m_nFeatures, 1, CV_64FC1);
//	//		Mat sigma(m_nFeatures, m_nFeatures, CV_64FC1);
//	//		
//	//		fread(&nPoints, sizeof(long), 1, pFile);
//	//		for (word y = 0; y < m_nFeatures; y++)
//	//			fread(&mu.at<double>(y, 0), sizeof(double), 1, pFile);
//	//		for (word y = 0; y < m_nFeatures; y++)
//	//			for (word x = 0; x < m_nFeatures; x++)
//	//				fread(&sigma.at<double>(y, x), sizeof(double), 1, pFile);
//	//		
//	//		gauss.setMu(mu);
//	//		gauss.setSigma(sigma);
//	//		gauss.setNumPoints(nPoints);
//	//		gauss.freeze();
//
//	//		mu.release();
//	//		sigma.release();
//	//	} // gausses
//	//} // s
//
//	//fread(&m_minCoefficient, sizeof(long double), 1, pFile);
//}
//
//void CTrainNodeGMM::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
//{
//	//Mat fv;
//	//Mat aux1, aux2, aux3;
//
//	//featureVector.convertTo(fv, CV_64FC1);
//
//	//for (byte s = 0; s < m_nStates; s++) {							// state
//	//	float	* pPot			= potential.ptr<float>(s);
//	//	byte	* pMask			= mask.ptr<byte>(s);
//	//
//	//	if (m_vGMs[s].empty()) {
//	//		// pPot[s] = 0;
//	//		pMask[s] = 0;
//	//		continue;
//	//	}
//
//	//	size_t nAllPoints	= 0;									// number of points were used for approximating the density for current state
//	//	for (auto &gauss : m_vGMs[s])
//	//		nAllPoints += gauss.getNumPoints();
//	//	
//	//	for (auto &gauss : m_vGMs[s]) {
//	//		double		k		= static_cast<double>(gauss.getNumPoints()) / nAllPoints;
//	//		double		value	= gauss.getValue(fv, aux1, aux2, aux3);
//	//		long double	aK		= gauss.getAlpha() / m_minCoefficient;					// scaled Gaussian coefficient
//	//		pPot[0] += static_cast<float>(k * aK * value);
//	//	} // gausses
//
//	//} // s
//}
//
//#ifdef DEBUG_PRINT_INFO			///@cond
//void printMat(const Mat &m)
//{
//	//for (register int y = 0; y < m.rows; y++) {
//	//	for(register int x = 0; x < m.cols; x++)
//	//		printf("%.1f\t", m.at<double>(y,x));
//	//	printf("\n");
//	//}
//}
//
//void CTrainNodeGMM::showStatus(void)
//{
//	//printf("\nCTrainNodeGMM::Status\n");
//	//printf("---------------------------\n");
//	//printf("( minCoefficient = %e )\n", m_minCoefficient);
//	//for (byte s = 0; s < m_nStates; s++) {		// states
//	//	printf("Class %d (%zu gausses):\n", s, m_vGMs[s].size());
//	//	word g = 0;
//	//	for (CNDGauss &gauss : m_vGMs[s]) {
//	//		printf("\tG[%u]: %zupts; ", g++,  gauss.getNumPoints());
//	//		printf("alpha: %e;\n", gauss.getAlpha());
//	//		//printf("aK: %e;\n", gauss.getAlpha() / m_minCoefficient);
//
//	//		Mat mu		= gauss.getMu();
//	//		Mat sigma	= gauss.getSigma();
//
//	//		//printf("mu =\n");		printMat(mu);
//	//		//printf("sigma =\n");	printMat(sigma); 
//	//		
//	//	} // gausses
//	//	printf("\n");
//	//} // s
//}
//#endif	// DEBUG_PRINT_INFO		///@endcond
//
//// ---------------------- Private functions ----------------------
//
//// Calculates distance from all Gaussians in a mixture to the point <x>
//// If when using Mahalanobis distance, a Gaussian is not full, the scaled Euclidian for this Gaussian is returned
//inline Mat CTrainNodeGMM::getDistance(byte s, const Mat &x) const
//{
//	//word	nGausses = static_cast<word>(m_vGMs[s].size());
//	//Mat		res(nGausses, 1, CV_64FC1);
//	//word g = 0;
//	//for (CNDGauss &gauss : m_vGMs[s]) {						// gausses
//	//	
//	//	if (m_params.dist_Mtreshold < 0) res.at<double>(g++, 0) = gauss.getEuclidianDistance(x);
//	//	else { 
//	//		if (gauss.getNumPoints() >= m_params.min_samples)	// if a Gaussian is full	
//	//			res.at<double>(g++, 0) = gauss.getMahalanobisDistance(x);
//	//		else {
//	//			double k = m_params.dist_Mtreshold / m_params.dist_Etreshold;
//	//			res.at<double>(g++, 0) = k * gauss.getEuclidianDistance(x);
//	//		}
//	//	} // mahalanobis
//	//} // gausses
//	//return res;
//}
//
//// Calculates divergence from Gaussian <x> to all other Gaussians in the mixture (including itself)
//// If a Gaussian from mixture is not full, the returned divergence is infinity
//// The caller should care about argument <x>
//inline Mat CTrainNodeGMM::getDivergence(byte s, CKDGauss *x) const
//{
//	return Mat();
//	//dword	nGausses = static_cast<dword>(m_vGMs[s].size());
//	//Mat		res(nGausses, 1, CV_64FC1, Scalar(DBL_MAX));
//	//word g = 0;
//	//for (CNDGauss &gauss : m_vGMs[s]) {
//	//	if (gauss.getNumPoints() >= m_params.min_samples)	// if the Gaussian is full
//	//		res.at<double>(g++, 0) = x->getKullbackLeiberDivergence(gauss);
//	//} // gausses
//	//return res;
//}


