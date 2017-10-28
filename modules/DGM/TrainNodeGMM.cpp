#include "TrainNodeGMM.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	// Constants
	const size_t		CTrainNodeGMM::MIN_SAMPLES = 16;
	const long double	CTrainNodeGMM::MAX_COEFFICIENT = 1.0;

	// Constructor
	CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures, TrainNodeGMMParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates), m_minCoefficient(1), m_params(params)
	{
		m_pvGausses = new GaussianMixture[nStates];
		for (int s = 0; s < nStates; s++)
			m_pvGausses[s].reserve(m_params.maxGausses);
		if (m_params.min_samples < MIN_SAMPLES) m_params.min_samples = MIN_SAMPLES;
	}

	// Constructor
	CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures, byte maxGausses) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates), m_minCoefficient(1), m_params(TRAIN_NODE_GMM_PARAMS_DEFAULT)
	{
		m_params.maxGausses = maxGausses;
		m_pvGausses = new GaussianMixture[nStates];
		for (int s = 0; s < nStates; s++)
			m_pvGausses[s].reserve(m_params.maxGausses);
	}

	// Destructor
	CTrainNodeGMM::~CTrainNodeGMM(void)
	{
		delete[] m_pvGausses;
	}

	void CTrainNodeGMM::reset(void)
	{
		for (byte s = 0; s < m_nStates; s++)
			m_pvGausses[s].clear();
		m_minCoefficient = 1;
	}

	void CTrainNodeGMM::addFeatureVec(const Mat &featureVector, byte gt)
	{
		Point	minPoint;

		// Assertions:
		DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %u is out of range [0; %u)", gt, m_nStates);

		Mat point(m_nFeatures, 1, CV_64FC1);
		featureVector.convertTo(point, point.type());

		if (m_pvGausses[gt].empty()) m_pvGausses[gt].emplace_back(point);	// NEW GAUSS
		else {
			Mat dist = getDistance(gt, point);								// Calculate distances all existing Gaussians in the mixture to the point

																			// Finding the smallest distance
			double minDist;
			minMaxLoc(dist, &minDist, NULL, &minPoint, NULL);
			int gaussIdx = minPoint.y;
			dist.release();

			double dist_treshold;
			if (m_params.dist_Mtreshold < 0)	dist_treshold = m_params.dist_Etreshold;
			else								dist_treshold = m_params.dist_Mtreshold;

			// Add to existing Gaussian or crete a new one
			if ((minDist > dist_treshold) && (m_pvGausses[gt].size() < m_params.maxGausses)) m_pvGausses[gt].emplace_back(point);		// NEW GAUSS
			else {
				CKDGauss *pGauss = &m_pvGausses[gt].at(gaussIdx);			// the nearest Gaussian
				if (false)	// TODO: check it
					pGauss->addPoint(point);								// update the nearest Gauss
				else
					*pGauss += CKDGauss(point);								// update the nearest Gauss

																			// Chech the updated Gauss function if after update it became too close to another Gauss function
				if ((m_params.div_KLtreshold > 0) && (pGauss->getNumPoints() >= m_params.min_samples)) {
					Mat div = getDivergence(gt, pGauss);
					div.at<double>(gaussIdx, 0) = DBL_MAX;					// divergence to itself

																			// Finding the smallest divergence
					double minDiv;
					minMaxLoc(div, &minDiv, NULL, &minPoint, NULL);
					int minGaussIdx = minPoint.y;
					div.release();

					// Merge together if they are too close
					if ((minGaussIdx >= 0) && (minDiv < m_params.div_KLtreshold)) {
						m_pvGausses[gt].at(minGaussIdx) += *pGauss;
						m_pvGausses[gt].erase(m_pvGausses[gt].begin() + gaussIdx);
					}
				}
			}
		}

		point.release();
	}

	void CTrainNodeGMM::train(bool)
	{
		register byte	s;

		// merge gausses with too small number of samples 
		for (s = 0; s < m_nStates; s++) {							// state
			for (auto it = m_pvGausses[s].begin(); it != m_pvGausses[s].end(); it++) {
				it->freeze();
				size_t nPoints = it->getNumPoints();
				if (nPoints < m_params.min_samples) {				// if Gaussian is not full
					word g = static_cast<word>(it - m_pvGausses[s].begin());
					if (nPoints >= MIN_SAMPLES) {
						Mat div = getDivergence(s, &(*it));
						div.at<double>(g, 0) = DBL_MAX;				// distance to itself (redundant here)

																	// Finding the smallest divergence
						double minDiv;
						Point minPoint;
						minMaxLoc(div, &minDiv, NULL, &minPoint, NULL);
						int gaussIdx = minPoint.y;
						div.release();
						if (gaussIdx >= 0) m_pvGausses[s].at(gaussIdx) += (*it);
					} // if Gaussian has less then MIN_SAMPLES points, consider it as a noise and delete
					m_pvGausses[s].erase(it);
					it--;
				} // if Gaussian Full
			} // gausses
		} // s

		  // getting the coefficients
		for (s = 0; s < m_nStates; s++) {				// state
			for (auto itGauss = m_pvGausses[s].begin(); itGauss != m_pvGausses[s].end(); itGauss++) {
				itGauss->freeze();
				long double Coefficient = itGauss->getAlpha();
				if (Coefficient > MAX_COEFFICIENT) {			// i.e. if (Coefficient = \infinitiy) delete Gaussian
					m_pvGausses[s].erase(itGauss);
					itGauss--;
					continue;
				}
				if (Coefficient < m_minCoefficient)
					m_minCoefficient = Coefficient;
			} // gausses
		} // s

#ifdef DEBUG_PRINT_INFO
		showStatus();
#endif	// DEBUG_PRINT_INFO
	}

	void CTrainNodeGMM::saveFile(FILE *pFile) const
	{
		// m_params
		fwrite(&m_params.maxGausses, sizeof(word), 1, pFile);
		fwrite(&m_params.min_samples, sizeof(size_t), 1, pFile);
		fwrite(&m_params.dist_Etreshold, sizeof(double), 1, pFile);
		fwrite(&m_params.dist_Mtreshold, sizeof(double), 1, pFile);
		fwrite(&m_params.div_KLtreshold, sizeof(double), 1, pFile);

		// m_pvpGausses;							
		for (byte s = 0; s < m_nStates; s++) {				// state
			word nGausses = static_cast<word>(m_pvGausses[s].size());
			fwrite(&nGausses, sizeof(word), 1, pFile);
			for (CKDGauss &gauss : m_pvGausses[s]) {
				size_t	nPoints = gauss.getNumPoints();
				Mat		mu = gauss.getMu();
				Mat		sigma = gauss.getSigma();

				fwrite(&nPoints, sizeof(long), 1, pFile);
				for (word y = 0; y < m_nFeatures; y++)
					fwrite(&mu.at<double>(y, 0), sizeof(double), 1, pFile);
				for (word y = 0; y < m_nFeatures; y++)
					for (word x = 0; x < m_nFeatures; x++)
						fwrite(&sigma.at<double>(y, x), sizeof(double), 1, pFile);
				mu.release();
				sigma.release();
			} // gauss
		} // s

		fwrite(&m_minCoefficient, sizeof(long double), 1, pFile);
	}

	void CTrainNodeGMM::loadFile(FILE *pFile)
	{
		// m_params
		fread(&m_params.maxGausses, sizeof(word), 1, pFile);
		fread(&m_params.min_samples, sizeof(size_t), 1, pFile);
		fread(&m_params.dist_Etreshold, sizeof(double), 1, pFile);
		fread(&m_params.dist_Mtreshold, sizeof(double), 1, pFile);
		fread(&m_params.div_KLtreshold, sizeof(double), 1, pFile);

		// m_pvpGausses;
		for (byte s = 0; s < m_nStates; s++) {				// state
			word nGausses;
			fread(&nGausses, sizeof(word), 1, pFile);
			m_pvGausses[s].assign(nGausses, CKDGauss(m_nFeatures));
			for (CKDGauss &gauss : m_pvGausses[s]) {
				long nPoints;
				Mat mu(m_nFeatures, 1, CV_64FC1);
				Mat sigma(m_nFeatures, m_nFeatures, CV_64FC1);

				fread(&nPoints, sizeof(long), 1, pFile);
				for (word y = 0; y < m_nFeatures; y++)
					fread(&mu.at<double>(y, 0), sizeof(double), 1, pFile);
				for (word y = 0; y < m_nFeatures; y++)
					for (word x = 0; x < m_nFeatures; x++)
						fread(&sigma.at<double>(y, x), sizeof(double), 1, pFile);

				gauss.setMu(mu);
				gauss.setSigma(sigma);
				gauss.setNumPoints(nPoints);
				gauss.freeze();

				mu.release();
				sigma.release();
			} // gausses
		} // s

		fread(&m_minCoefficient, sizeof(long double), 1, pFile);
	}

	void CTrainNodeGMM::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
		Mat fv;
		Mat aux1, aux2, aux3;

		featureVector.convertTo(fv, CV_64FC1);

		for (byte s = 0; s < m_nStates; s++) {							// state
			float	* pPot = potential.ptr<float>(s);
			byte	* pMask = mask.ptr<byte>(s);

			if (m_pvGausses[s].empty()) {
				// pPot[s] = 0;
				pMask[s] = 0;
				continue;
			}

			size_t nAllPoints = 0;									// number of points were used for approximating the density for current state
			for (CKDGauss &gauss : m_pvGausses[s])
				nAllPoints += gauss.getNumPoints();

			for (CKDGauss &gauss : m_pvGausses[s]) {
				double		k = static_cast<double>(gauss.getNumPoints()) / nAllPoints;
				double		value = gauss.getValue(fv, aux1, aux2, aux3);
				long double	aK = gauss.getAlpha() / m_minCoefficient;					// scaled Gaussian coefficient
				pPot[0] += static_cast<float>(k * aK * value);
			} // gausses

		} // s
	}

#ifdef DEBUG_PRINT_INFO			///@cond
	void printMat(Mat &m)
	{
		for (register int y = 0; y < m.rows; y++) {
			for (register int x = 0; x < m.cols; x++)
				printf("%.1f\t", m.at<double>(y, x));
			printf("\n");
		}
	}

	void CTrainNodeGMM::showStatus(void)
	{
		printf("\nCTrainNodeGMM::Status\n");
		printf("---------------------------\n");
		printf("( minCoefficient = %e )\n", m_minCoefficient);
		for (byte s = 0; s < m_nStates; s++) {		// states
			printf("Class %d (%zu gausses):\n", s, m_pvGausses[s].size());
			word g = 0;
			for (CKDGauss &gauss : m_pvGausses[s]) {
				printf("\tG[%u]: %zupts; ", g++, gauss.getNumPoints());
				printf("alpha: %e;\n", gauss.getAlpha());
				//printf("aK: %e;\n", gauss.getAlpha() / m_minCoefficient);

				Mat mu = gauss.getMu();
				Mat sigma = gauss.getSigma();

				//printf("mu =\n");		printMat(mu);
				//printf("sigma =\n");	printMat(sigma); 

			} // gausses
			printf("\n");
		} // s
	}
#endif	// DEBUG_PRINT_INFO		///@endcond

	// ---------------------- Private functions ----------------------

	// Calculates distance from all Gaussians in a mixture to the point <x>
	// If when using Mahalanobis distance, a Gaussian is not full, the scaled Euclidian for this Gaussian is returned
	inline Mat CTrainNodeGMM::getDistance(byte s, const Mat &x) const
	{
		word	nGausses = static_cast<word>(m_pvGausses[s].size());
		Mat		res(nGausses, 1, CV_64FC1);
		word g = 0;
		for (CKDGauss &gauss : m_pvGausses[s]) {						// gausses

			if (m_params.dist_Mtreshold < 0) res.at<double>(g++, 0) = gauss.getEuclidianDistance(x);
			else {
				if (gauss.getNumPoints() >= m_params.min_samples)	// if a Gaussian is full	
					res.at<double>(g++, 0) = gauss.getMahalanobisDistance(x);
				else {
					double k = m_params.dist_Mtreshold / m_params.dist_Etreshold;
					res.at<double>(g++, 0) = k * gauss.getEuclidianDistance(x);
				}
			} // mahalanobis
		} // gausses
		return res;
	}

	// Calculates divergence from Gaussian <x> to all other Gaussians in the mixture (including itself)
	// If a Gaussian from mixture is not full, the returned divergence is infinity
	// The caller should care about argument <x>
	inline Mat CTrainNodeGMM::getDivergence(byte s, CKDGauss *x) const
	{
		dword	nGausses = static_cast<dword>(m_pvGausses[s].size());
		Mat		res(nGausses, 1, CV_64FC1, Scalar(DBL_MAX));
		word g = 0;
		for (CKDGauss &gauss : m_pvGausses[s]) {
			if (gauss.getNumPoints() >= m_params.min_samples)	// if the Gaussian is full
				res.at<double>(g++, 0) = x->getKullbackLeiberDivergence(gauss);
		} // gausses
		return res;
	}
}

namespace DirectGraphicalModels_Aux
{
	// Constants
	const size_t CTrainNodeGMM::MIN_SAMPLES		= 16;
	
	// Constructor
	CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures, TrainNodeGMMParams params)
		: CTrainNode(nStates, nFeatures)
		, CBaseRandomModel(nStates)
		, m_params(params)
	{
		m_vGaussianMixtures.resize(nStates);
		for (auto &gaussianMixture : m_vGaussianMixtures)
			gaussianMixture.reserve(m_params.maxGausses);
	}
	
	// Constructor
	CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures, word maxGausses) 
		: CTrainNode(nStates, nFeatures) 
		, CBaseRandomModel(nStates)
	{
		m_params.maxGausses = maxGausses;
		m_vGaussianMixtures.resize(nStates);
		for (auto &gaussianMixture : m_vGaussianMixtures)
			gaussianMixture.reserve(m_params.maxGausses);
	}

	void CTrainNodeGMM::reset(void) {
		m_vGaussianMixtures.clear();
	}
	
	namespace {
		// Calculates distance from all Gaussians in a mixture to the point <x>
		// If when using Mahalanobis distance, a Gaussian is not full, the scaled Euclidian for this Gaussian is returned
		inline std::vector<double> getDistance(const Mat &x, const GaussianMixture &gaussianMixture, size_t samplesTreshold, double dist_Etreshold, double dist_Mtreshold)
		{
			std::vector<double> res(gaussianMixture.size());
			for (size_t i = 0; i < res.size(); i++)
				if (dist_Mtreshold)		// Euclidian distance
					res[i] = gaussianMixture[i].getEuclidianDistance(x);
				else 					// Mahalanobis distance
					res[i] = gaussianMixture[i].getNumPoints() >= samplesTreshold ? gaussianMixture[i].getMahalanobisDistance(x)
																				  : gaussianMixture[i].getEuclidianDistance(x) * dist_Mtreshold / dist_Etreshold;
			return res;
		}
		
		// Calculates divergence from Gaussian <x> to all other Gaussians in the mixture (including itself)
		// If a Gaussian from mixture is not full, the returned divergence is infinity
		// The caller should care about argument <x>
		inline std::vector<double> getDivergence(const CKDGauss &x, const GaussianMixture &gaussianMixture, size_t samplesTreshold)
		{
			std::vector<double> res(gaussianMixture.size());
			for (size_t i = 0; i < res.size(); i++)
				res[i] = gaussianMixture[i].getNumPoints() >= samplesTreshold ? x.getKullbackLeiberDivergence(gaussianMixture[i]) : DBL_MAX;
			return res;
		}
	}

	void CTrainNodeGMM::addFeatureVec(const Mat &featureVector, byte gt) {
		Mat point;
		featureVector.convertTo(point, CV_64FC1);

		GaussianMixture &gaussianMixture = m_vGaussianMixtures[gt];						// GMM of current state		

		if (gaussianMixture.empty()) gaussianMixture.emplace_back(point);				// NEW GAUSS
		else {
			// Find the nearest gaussian distribution 
			// Calculate distances between all existing Gaussians in the mixture to the point
			std::vector<double> dist = getDistance(point, gaussianMixture, m_params.min_samples, m_params.dist_Etreshold, m_params.dist_Mtreshold);

			// Find the smallest distance
			auto it = std::min_element(dist.begin(), dist.end());
			double minDist = *it;
			
			double dist_treshold = (m_params.dist_Mtreshold < 0) ? m_params.dist_Etreshold : m_params.dist_Mtreshold;

			// Add to existing Gaussian or crete a new one
			if (gaussianMixture.size() < m_params.maxGausses && minDist > dist_treshold) 
				gaussianMixture.emplace_back(point);									// NEW GAUSS
			else {
				// Add to existing Gaussian
				size_t updIdx = std::distance(dist.begin(), it);
				CKDGauss &updGauss = gaussianMixture[updIdx];							// the nearest Gaussian
				updGauss.addPoint(point, false);												// update the nearest Gauss

				// Check if the updated Gauss function became too close to another Gauss function
				if (m_params.div_KLtreshold > 0 && updGauss.getNumPoints() >= m_params.min_samples) {
					// Calculate divergences between updGauss and all other gausses
					std::vector<double> div = getDivergence(updGauss, gaussianMixture, m_params.min_samples);
					div[updIdx] = DBL_MAX;						// divergence to itself

					// Find the smallest divergence
					auto it = std::min_element(div.begin(), div.end());
					double minDivg = *it;
					size_t idx = std::distance(div.begin(), it);

					// Merge together if they are too close
					if (minDivg < m_params.div_KLtreshold) {
						gaussianMixture[idx] += updGauss;
						gaussianMixture.erase(gaussianMixture.begin() + updIdx);
					}
				}
			}
		}
	}

	namespace {
		template<typename T>
		void printMat(const std::string &name, const Mat &m) {
			printf("%s:\n", name.c_str());
			for (int y = 0; y < m.rows; y++) {
				for (int x = 0; x < m.cols; x++)
					printf("%.1f\t", m.at<T>(y, x));
				printf("\n");
			}
		}
	
		void printStatus(std::vector<GaussianMixture> &vGaussianMixtures) {
			printf("\nCTrainNodeGMM::Status\n");
			printf("---------------------------\n");
//			printf("( minCoefficient = %e )\n", m_minCoefficient);
	
			for (size_t s = 0; s < vGaussianMixtures.size(); s++) {		// states
				printf("Class %zu (%zu gausses):\n", s, vGaussianMixtures[s].size());
				
				word g = 0;
				for (const CKDGauss &gauss : vGaussianMixtures[s]) {
					printf("\tG[%u]: %zupts; ", g++, gauss.getNumPoints());
					printf("alpha: %e;\n", gauss.getAlpha());
					//printf("aK: %e;\n", gauss.getAlpha() / m_minCoefficient);

					//printMat<double>("MU:", gauss.getMu());
					//printMat<double>("SIGMA:", gauss.getSigma()); 

				} // gausses
				printf("\n");
			} // s
		}
	}

	void CTrainNodeGMM::train(bool doClean) {
	
		// merge gausses with too small number of samples 
		for (GaussianMixture &gaussianMixture : m_vGaussianMixtures) {							// state
			for (auto it = gaussianMixture.begin(); it != gaussianMixture.end(); it++) {
				//			it->freeze();
				size_t nPoints = it->getNumPoints();
				if (nPoints < m_params.min_samples) {				// if Gaussian is not full
					if (nPoints >= MIN_SAMPLES) {
						size_t g = std::distance(gaussianMixture.begin(), it);
						std::vector<double> div = getDivergence(*it, gaussianMixture, m_params.min_samples);
						div[g] = DBL_MAX;							// distance to itself (redundant here)

						// Finding the smallest divergence
						size_t gaussIdx = std::distance(div.begin(), std::min_element(div.begin(), div.end()));
						gaussianMixture[gaussIdx] += (*it);
					} // if Gaussian has less then MIN_SAMPLES points, consider it as a noise and delete
					gaussianMixture.erase(it);
					it--;
				} // if Gaussian Full
			} // gausses
		} // s

		// getting the coefficients
		//for (s = 0; s < m_nStates; s++) {				// state
		//	for (auto itGauss = m_vGMs[s].begin(); itGauss != m_vGMs[s].end(); itGauss++) {
		//		itGauss->freeze();
		//		long double Coefficient = itGauss->getAlpha();
		//		if (Coefficient > MAX_COEFFICIENT) {			// i.e. if (Coefficient = \infinitiy) delete Gaussian
		//			m_vGMs[s].erase(itGauss);
		//			itGauss--;
		//			continue;
		//		}
		//		if (Coefficient < m_minCoefficient)
		//			m_minCoefficient = Coefficient;
		//	} // gausses
		//} // s

		//#ifdef DEBUG_PRINT_INFO
			printStatus(m_vGaussianMixtures);
		//#endif	// DEBUG_PRINT_INFO
	}

	void CTrainNodeGMM::saveFile(FILE *pFile) const {}
	
	void CTrainNodeGMM::loadFile(FILE *pFile) {}

	void CTrainNodeGMM::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const {}
}

