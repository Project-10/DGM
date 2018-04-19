// k-Nearest Neighbors (based on OpenCV) training class interface
// Written by Sergey G. Kosov in 2017 for Project X
#pragma once

#include "TrainNode.h"

namespace DirectGraphicalModels
{
	class CSamplesAccumulator;

	///@brief OpenCV k-Nearest Neighbors parameters
	typedef struct TrainNodeCvKNNParams {
		float	bias;								///< Regularization CRF parameter: bias is added to all potential values
		size_t	maxNeighbors;						///< Max number of neighbors to be used for calculating potentials
		size_t 	maxSamples;							///< Maximum number of samples to be used in training. 0 means using all the samples

		TrainNodeCvKNNParams() {}
		TrainNodeCvKNNParams(float _bias, size_t _maxNeighbors, size_t _maxSamples) : bias(_bias), maxNeighbors(_maxNeighbors), maxSamples(_maxSamples) {}
	} TrainNodeCvKNNParams;

	const TrainNodeCvKNNParams TRAIN_NODE_CV_KNN_PARAMS_DEFAULT =	TrainNodeCvKNNParams(
																	0.1f,	// Regularization CRF parameter: bias is added to all potential values
																	100,	// Max number of neighbors to be used for calculating potentials
																	0		// Maximum number of samples to be used in training. 0 means using all the samples
																	);

	// ====================== OpenCV k-Nearest Neighbors Train Class =====================
	/**
	* @ingroup moduleTrainNode
	* @brief OpenCV Nearest Neighbor training class
	* @details This class implements the <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm" target="blank"><i>k</i>-nearest neighbors classifier (<i>k</i>-NN)</a>,
	* where the input consists of the k closest training samples in the feature space and the output depends on k-Nearest Neighbors.
	* > This trainer is especially effective for low-dimentional feature spaces.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeCvKNN : public CTrainNode {
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params k-Nearest Neighbors parameters (Ref. @ref TrainNodeCvKNNParams)
		*/
		DllExport CTrainNodeCvKNN(byte nStates, word nFeatures, TrainNodeCvKNNParams params = TRAIN_NODE_CV_KNN_PARAMS_DEFAULT);
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxSamples Maximum number of samples to be used in training
		* > Default value \b 0 means using all the samples.<br>
		* > If another value is specified, the class for training will use \b maxSamples random samples from the whole amount of samples, added via addFeatureVec() function
		*/
		CTrainNodeCvKNN(byte nStates, word nFeatures, size_t maxSamples);
		DllExport virtual ~CTrainNodeCvKNN(void);

		DllExport void	reset(void);
		DllExport void	save(const std::string &path, const std::string &name = std::string(), short idx = -1) const;
		DllExport void	load(const std::string &path, const std::string &name = std::string(), short idx = -1);

		DllExport void	addFeatureVec(const Mat &featureVector, byte gt);

		DllExport void	train(bool doClean = false);


	protected:
		DllExport void	saveFile(FILE *pFile) const { }
		DllExport void	loadFile(FILE *pFile) { }
		DllExport void  calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const;

	
	protected:
		Ptr<ml::KNearest>		  m_pKNN;					///< k-Nearest Neighbors
		CSamplesAccumulator		* m_pSamplesAcc;			///< Samples Accumulator
	

	private:
		void		  init(TrainNodeCvKNNParams params);		// This function is called by both constructors


	private:
		TrainNodeCvKNNParams	  m_params;
	};
}