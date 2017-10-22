// Gaussian Mixture Model training class interface
// Written by Sergey G. Kosov in 2012 - 2014, 2017 for Project X
#pragma once

#include "TrainNode.h"
#include "KDGauss.h"

namespace DirectGraphicalModels {
	///@brief Gaussian Mixture Model parameters
	struct TrainNodeGMMParams {
		word	maxGausses		= 64;				///< The maximal number of Gauss functions for approximation
		size_t	min_samples		= 64;				///< Minimum number of sapmles to approximate a Gauss function		
		double	dist_Etreshold	= 64;				///< Minimum Euclidean distance between Gauss functions
		double	dist_Mtreshold	= -16;				///< Minimum Mahalanobis distance between Gauss functions. If this parameter is negative, the Euclidean distance is used
		double	div_KLtreshold	= -16;				///< Minimum Kullback-Leiber divergence between Gauss functions. If this parameter is negative, the merging of Gaussians in addFeatureVec() function will be disabled
	};
	

	class CTrainNodeGMM : public CTrainNode {
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params Gaussian Mixture Model parameters (Ref. @ref TrainNodeGMMParams)
		*/
		DllExport CTrainNodeGMM(byte nStates, word nFeatures, TrainNodeGMMParams params = TrainNodeGMMParams());
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxGausses The maximal number of mixture components in the Gaussian Mixture Model per state (class)
		*/
		DllExport CTrainNodeGMM(byte nStates, word nFeatures, word maxGausses);

		DllExport virtual ~CTrainNodeGMM(void) {}

		DllExport virtual void	reset(void) override;	
		DllExport virtual void	addFeatureVec(const Mat &featureVector, byte gt) override;
		DllExport virtual void	train(bool doClean = false) override;
	

	protected:
		DllExport virtual void	saveFile(FILE *pFile) const override; 
		DllExport virtual void	loadFile(FILE *pFile) override; 
		DllExport virtual void	calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const override;


	protected:
		std::vector<GaussianMixture>	m_vGaussianMixtures;


	private:
		static const size_t				MIN_SAMPLES;


	private: 
		TrainNodeGMMParams				m_params;
	};
}
