#include "TrainNodeGMM.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	// Constants
	const size_t		CTrainNodeGMM::MIN_SAMPLES = 16;
	const long double	CTrainNodeGMM::MAX_COEFFICIENT = 1.0;

	// Constructor
	CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures, TrainNodeGMMParams params) 
        : CBaseRandomModel(nStates)
        , CTrainNode(nStates, nFeatures)
		, m_params(params)
	{
		m_vGaussianMixtures.resize(nStates);
		for (auto &gaussianMixture : m_vGaussianMixtures)
			gaussianMixture.reserve(m_params.maxGausses);
		if (m_params.minSamples < MIN_SAMPLES) m_params.minSamples = MIN_SAMPLES;
	}

	// Constructor
	CTrainNodeGMM::CTrainNodeGMM(byte nStates, word nFeatures, byte maxGausses)
        : CBaseRandomModel(nStates)
        , CTrainNode(nStates, nFeatures)
        , m_params(TRAIN_NODE_GMM_PARAMS_DEFAULT)
	{
		m_params.maxGausses = maxGausses;
		m_vGaussianMixtures.resize(nStates);
		for (auto &gaussianMixture : m_vGaussianMixtures)
			gaussianMixture.reserve(m_params.maxGausses);
	}

	void CTrainNodeGMM::reset(void)
	{
		m_vGaussianMixtures.clear();
		m_minAlpha = 1;
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
		// Assertions
		DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %u is out of range [0; %u)", gt, m_nStates);

		Mat point;
		featureVector.convertTo(point, CV_64FC1);

		GaussianMixture &gaussianMixture = m_vGaussianMixtures[gt];							// GMM of current state		

		if (gaussianMixture.empty()) 
			gaussianMixture.emplace_back(point);			// NEW GAUSS
		else {
			std::vector<double> dist = getDistance(point, gaussianMixture, m_params.minSamples, m_params.dist_Etreshold, m_params.dist_Mtreshold);					// Calculate distances all existing Gaussians in the mixture to the point

			// Find the smallest distance
			auto it = std::min_element(dist.begin(), dist.end());
			double minDist = *it;

			double dist_treshold = (m_params.dist_Mtreshold < 0) ? m_params.dist_Etreshold : m_params.dist_Mtreshold;

			// Add to existing Gaussian or crete a new one
			if ((minDist > dist_treshold) && (gaussianMixture.size() < m_params.maxGausses)) 
				gaussianMixture.emplace_back(point);		// NEW GAUSS
			else {
				size_t updIdx = std::distance(dist.begin(), it);
				CKDGauss &updGauss = gaussianMixture[updIdx];				// the nearest Gaussian
				updGauss += point;											// update the nearest Gauss

				// Chech the updated Gauss function if after update it became too close to another Gauss function
				if ((m_params.div_KLtreshold > 0) && (updGauss.getNumPoints() >= m_params.minSamples)) {
					// Calculate divergences between updGauss and all other gausses
					std::vector<double> div = getDivergence(updGauss, gaussianMixture, m_params.minSamples);
					div[updIdx] = DBL_MAX;									// divergence to itself

					// Find the smallest divergence
					auto it = std::min_element(div.begin(), div.end());

					// Merge together if they are too close
					if ((it != div.end()) && (*it < m_params.div_KLtreshold)) {
						size_t idx = std::distance(div.begin(), it);
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
			if (!name.empty()) printf("%s:\n", name.c_str());
			for (int y = 0; y < m.rows; y++) {
				for (int x = 0; x < m.cols; x++)
					printf("%.1f\t", m.at<T>(y, x));
				printf("\n");
			}
		}

		void printStatus(std::vector<GaussianMixture> &vGaussianMixtures, long double minCoefficient) {
#ifdef DEBUG_PRINT_INFO
            printf("\nCTrainNodeGMM::Status\n");
			printf("---------------------------\n");
			printf("( minCoefficient = %Le )\n", minCoefficient);

			for (size_t s = 0; s < vGaussianMixtures.size(); s++) {		// states
				printf("Class %zu (%zu gausses):\n", s, vGaussianMixtures[s].size());

				word g = 0;
				for (const CKDGauss &gauss : vGaussianMixtures[s]) {
					printf("\tG[%u]: %zupts; ", g++, gauss.getNumPoints());
					printf("alpha: %Le;\n", gauss.getAlpha());
					//printf("aK: %e;\n", gauss.getAlpha() / m_minCoefficient);

					//printMat<double>("MU:", gauss.getMu());
					//printMat<double>("SIGMA:", gauss.getSigma()); 
				} // gausses
				printf("\n");
			} // s
#endif
		}
	}

	void CTrainNodeGMM::train(bool)
	{
		// merge gausses with too small number of samples 
		for (GaussianMixture &gaussianMixture : m_vGaussianMixtures) {			// state
			for (auto it = gaussianMixture.begin(); it != gaussianMixture.end(); it++) {
				size_t nPoints = it->getNumPoints();
				if (nPoints < m_params.minSamples) {				// if Gaussian is not full
					if (nPoints >= MIN_SAMPLES) {
						size_t g = std::distance(gaussianMixture.begin(), it);
						std::vector<double> div = getDivergence(*it, gaussianMixture, m_params.minSamples);
						div[g] = DBL_MAX;							// distance to itself (redundant here)

						// Finding the smallest divergence
						auto itm = std::min_element(div.begin(), div.end());
						if (itm != div.end()) {
							size_t gaussIdx = std::distance(div.begin(), itm);
							gaussianMixture[gaussIdx] += (*it);
						}
					} // if Gaussian has less then MIN_SAMPLES points, consider it as a noise and delete
					gaussianMixture.erase(it);
					it--;
				} // if Gaussian full
			} // gausses
		} // gaussianMixture

		// getting the coefficients
		for (GaussianMixture &gaussianMixture : m_vGaussianMixtures) {			// state
			for (auto itGauss = gaussianMixture.begin(); itGauss != gaussianMixture.end(); itGauss++) {
				long double alpha = itGauss->getAlpha();
				if (alpha > MAX_COEFFICIENT) {			// i.e. if (Coefficient = \infinitiy) delete Gaussian
					gaussianMixture.erase(itGauss);
					itGauss--;
					continue;
				}
				if (m_minAlpha > alpha)
					m_minAlpha = alpha;
			} // gausses
		} // gaussianMixture

		printStatus(m_vGaussianMixtures, m_minAlpha);
	}

	void CTrainNodeGMM::saveFile(FILE *pFile) const
	{
		// m_params
		fwrite(&m_params.maxGausses, sizeof(word), 1, pFile);
		fwrite(&m_params.minSamples, sizeof(size_t), 1, pFile);
		fwrite(&m_params.dist_Etreshold, sizeof(double), 1, pFile);
		fwrite(&m_params.dist_Mtreshold, sizeof(double), 1, pFile);
		fwrite(&m_params.div_KLtreshold, sizeof(double), 1, pFile);

		// m_pvpGausses;							
		for (const GaussianMixture &gaussianMixture : m_vGaussianMixtures) {	// state
			word nGausses = static_cast<word>(gaussianMixture.size());
			fwrite(&nGausses, sizeof(word), 1, pFile);
			for (const CKDGauss &gauss : gaussianMixture) {
				size_t	nPoints = gauss.getNumPoints();
				Mat		mu		= gauss.getMu();
				Mat		sigma	= gauss.getSigma();

				fwrite(&nPoints, sizeof(long), 1, pFile);
				for (word y = 0; y < getNumFeatures(); y++)
					fwrite(&mu.at<double>(y, 0), sizeof(double), 1, pFile);
				for (word y = 0; y < getNumFeatures(); y++)
					for (word x = 0; x < getNumFeatures(); x++)
						fwrite(&sigma.at<double>(y, x), sizeof(double), 1, pFile);
				mu.release();
				sigma.release();
			} // gauss
		} // gaussianMixture

		fwrite(&m_minAlpha, sizeof(long double), 1, pFile);
	}

	void CTrainNodeGMM::loadFile(FILE *pFile)
	{
		// m_params
		fread(&m_params.maxGausses, sizeof(word), 1, pFile);
		fread(&m_params.minSamples, sizeof(size_t), 1, pFile);
		fread(&m_params.dist_Etreshold, sizeof(double), 1, pFile);
		fread(&m_params.dist_Mtreshold, sizeof(double), 1, pFile);
		fread(&m_params.div_KLtreshold, sizeof(double), 1, pFile);

		// m_pvpGausses;
		m_vGaussianMixtures.resize(m_nStates);
		for (GaussianMixture &gaussianMixture : m_vGaussianMixtures) {				// state
			word nGausses;
			fread(&nGausses, sizeof(word), 1, pFile);
			gaussianMixture.assign(nGausses, CKDGauss(getNumFeatures()));
			for (CKDGauss &gauss : gaussianMixture) {
				long nPoints;
				Mat mu(getNumFeatures(), 1, CV_64FC1);
				Mat sigma(getNumFeatures(), getNumFeatures(), CV_64FC1);

				fread(&nPoints, sizeof(long), 1, pFile);
				for (word y = 0; y < getNumFeatures(); y++)
					fread(&mu.at<double>(y, 0), sizeof(double), 1, pFile);
				for (word y = 0; y < getNumFeatures(); y++)
					for (word x = 0; x < getNumFeatures(); x++)
						fread(&sigma.at<double>(y, x), sizeof(double), 1, pFile);

				gauss.setMu(mu);
				gauss.setSigma(sigma);
				gauss.setNumPoints(nPoints);

				mu.release();
				sigma.release();
			} // gausses
		} // gaussianMixture

		fread(&m_minAlpha, sizeof(long double), 1, pFile);
	}

	void CTrainNodeGMM::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
	{
		Mat fv;
		Mat aux1, aux2, aux3;

		featureVector.convertTo(fv, CV_64FC1);

		for (byte s = 0; s < m_nStates; s++) {						// state
			const GaussianMixture &gaussianMixture = m_vGaussianMixtures[s];

			if (gaussianMixture.empty())	mask.at<byte>(s, 0) = 0;
			else {
				size_t nAllPoints = 0;									// number of points were used for approximating the density for current state
				for (const CKDGauss &gauss : gaussianMixture)
					nAllPoints += gauss.getNumPoints();

				for (const CKDGauss &gauss : gaussianMixture) {
					double		k = static_cast<double>(gauss.getNumPoints()) / nAllPoints;
					double		value = gauss.getValue(fv, aux1, aux2, aux3);
					long double	aK = gauss.getAlpha() / m_minAlpha;		// scaled Gaussian coefficient
					potential.at<float>(s, 0) += static_cast<float>(k * aK * value);
				} // gausses
			}
		} // s
	}
}
