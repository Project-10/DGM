#include "TrainNodeGMM.h"
#include "macroses.h"

namespace DirectGraphicalModels
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

