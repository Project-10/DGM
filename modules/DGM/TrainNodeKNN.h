// k-Nearest Neighbors training class interface
// Written by Sergey G. Kosov in 2017 for Project X
#pragma once

#include "TrainNode.h"

namespace DirectGraphicalModels
{
	class CKDTree;
	class CSamplesAccumulator;

	/// @brief k-Nearest Neighbors parameters
	typedef struct TrainNodeKNNParams {
		float	bias;								///< Regularization CRF parameter: bias is added to all potential values
		size_t	maxNeighbors;						///< Max number of neighbors to be used for calculating potentials
		size_t 	maxSamples;							///< Maximum number of samples to be used in training. 0 means using all the samples

		TrainNodeKNNParams() {}
		TrainNodeKNNParams(float _bias, size_t _maxNeighbors, size_t _maxSamples) : bias(_bias), maxNeighbors(_maxNeighbors), maxSamples(_maxSamples) {}
	} TrainNodeKNNParams;
	
	const TrainNodeKNNParams TRAIN_NODE_KNN_PARAMS_DEFAULT =	TrainNodeKNNParams(
																0.1,	// Regularization CRF parameter: bias is added to all potential values
																100,	// Max number of neighbors to be used for calculating potentials
																0		// Maximum number of samples to be used in training. 0 means using all the samples
																);

	// ====================== k-Nearest Neighbors Train Class =====================
	/**
	* @ingroup moduleTrainNode
	* @brief Nearest Neighbor training class
	* @details This class implements the <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm" target="blank">k-nearest neighbors classifier (k-NN)</a>,
	* where the input consists of the k closest training samples in the feature space and the output depends on k-Nearest Neighbors.
	* > This trainer is especially effective for low-dimentional feature spaces.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeKNN : public CTrainNode
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params k-Nearest Neighbors parameters (Ref. @ref TrainNodeKNNParams)
		*/
		DllExport CTrainNodeKNN(byte nStates, word nFeatures, TrainNodeKNNParams params = TRAIN_NODE_KNN_PARAMS_DEFAULT);
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxSamples Maximum number of samples to be used in training.
		* > Default value \b 0 means using all the samples.<br>
		* > If another value is specified, the class for training will use \b maxSamples random samples from the whole amount of samples, added via addFeatureVec() function
		*/
		DllExport CTrainNodeKNN(byte nStates, word nFeatures, size_t maxSamples);
		DllExport ~CTrainNodeKNN(void);

		DllExport void	reset(void);
		DllExport void	save(const std::string &path, const std::string &name = std::string(), short idx = -1) const;
		DllExport void	load(const std::string &path, const std::string &name = std::string(), short idx = -1);

		DllExport void	addFeatureVec(const Mat &featureVector, byte gt);
		DllExport void	train(bool doClean = false);


	protected:
		DllExport void	saveFile(FILE *pFile) const {}
		DllExport void	loadFile(FILE *pFile) {}
		DllExport void	calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const;


	protected:
		CKDTree				* m_pTree;
		CSamplesAccumulator * m_pSamplesAcc;


	private:
		void					init(TrainNodeKNNParams params);	// This function is called by both constructors


	private:
		TrainNodeKNNParams	  m_params;
	};
}