// Gaussian Mixture Model training class interface
// Written by Sergey G. Kosov in 2012 - 2014 for Project X
// Refactored by Sergey G. Kosov in 2017 for Project X
#pragma once

#include "TrainNode.h"
#include "KDGauss.h"

namespace DirectGraphicalModels
{
	///@brief Gaussian Mixture Model parameters
	typedef struct TrainNodeGMMParams {
		word	maxGausses;					///< The maximal number of Gauss functions for approximation
		size_t	minSamples;					///< Minimum number of sapmles to approximate a Gauss function
		double	dist_Etreshold;				///< Minimum Euclidean distance between Gauss functions
		double	dist_Mtreshold;				///< Minimum Mahalanobis distance between Gauss functions. If this parameter is negative, the Euclidean distance is used
		double	div_KLtreshold;				///< Minimum Kullback-Leiber divergence between Gauss functions. If this parameter is negative, the merging of Gaussians in addFeatureVec() function will be disabled

		TrainNodeGMMParams() {}
		TrainNodeGMMParams(word _maxGausses, size_t _minSamples, double _dist_Etreshold, double _dist_Mtreshold, double _div_KLtreshold) : maxGausses(_maxGausses), minSamples(_minSamples), dist_Etreshold(_dist_Etreshold), dist_Mtreshold(_dist_Mtreshold), div_KLtreshold(_div_KLtreshold) {}
	} TrainNodeGMMParams;

	const TrainNodeGMMParams TRAIN_NODE_GMM_PARAMS_DEFAULT = TrainNodeGMMParams(
		64,		// maxGausses
		64,		// min_samples
		64,		// dist_Etreshold
		-16,	// dist_Mtreshold
		-16		// div_KLtreshold
	);

	// ==================== Gaussian Mixture Model Train Class =====================
	/**
	* @ingroup moduleTrainNode
	* @brief Gaussian Mixture Model training class
	* @details This class implements the generative training mechanism, based on the idea of approximating the density of multi-dimensional random variables
	* with an additive super-position of multivariate Gaussian distributions. The underlying algorithm is described in the paper
	* <a href="http://www.project-10.de/Kosov/files/GCPR_2013.pdf" target="_blank">Sequential Gaussian Mixture Models for Two-Level Conditional Random Fields</a>
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeGMM : public CTrainNode
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params Gaussian Mixture Model parameters (Ref. @ref TrainNodeGMMParams)
		*/
		DllExport CTrainNodeGMM(byte nStates, word nFeatures, TrainNodeGMMParams params = TRAIN_NODE_GMM_PARAMS_DEFAULT);
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxGausses The maximal number of mixture components in the Gaussian Mixture Model per state (class)
		*/
		DllExport CTrainNodeGMM(byte nStates, word nFeatures, byte maxGausses);
		DllExport virtual ~CTrainNodeGMM(void) = default;

		DllExport void	reset(void);

		DllExport void	addFeatureVec(const Mat &featureVector, byte gt);
		DllExport void	train(bool doClean = false);


	protected:
		DllExport void	saveFile(FILE *pFile) const;
		DllExport void	loadFile(FILE *pFile);
		/**
		* @brief Calculates the node potential, based on the feature vector
		* @details This function calculates the potentials of the node, described with the sample \a featureVector (\f$ \textbf{f} \f$):
		* \f$ nodePot_s = \sum^{nGaussians_s}_{i=1}\pi_{i,s}\cdot\mathcal{N}_{i,s}(\textbf{f}), \forall s \in \mathbb{S} \f$, where \f$\mathbb{S}\f$ is the set of all states (classes) and \f$\pi\f$ is a weighted coefficient.
		* In other words, the indexes: \f$ s \in [0; nStates) \f$. Here \f$ \mathcal{N} \f$ is a Gaussian function kernel, described in class @ref CKDGauss
		* @param[in]	featureVector Multi-dimensinal point \f$\textbf{f}\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1)
		* @param[in,out]	potential %Node potentials: Mat(size: nStates x 1; type: CV_32FC1). This parameter should be preinitialized and set to value 0.
		* @param[in,out]	mask Relevant %Node potentials: Mat(size: nStates x 1; type: CV_8UC1). This parameter should be preinitialized and set to value 1 (all potentials are relevant).
		*/
		DllExport void calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const;


	private:
		static const size_t				MIN_SAMPLES;
		static const long double		MAX_COEFFICIENT;


	private:
		TrainNodeGMMParams				m_params;
		std::vector<GaussianMixture>	m_vGaussianMixtures;						// block of n-dimensional Gauss function	
		long double						m_minAlpha = 1;								// auxilary coefficient for scaling gaussian coefficients
	};
}


