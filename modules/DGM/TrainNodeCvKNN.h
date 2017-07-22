// k-Nearest Neighbors (based on OpenCV) training class interface
// Written by Sergey G. Kosov in 2017 for Project X
#pragma once

#include "TrainNode.h"

namespace DirectGraphicalModels
{
	class CSamplesAccumulator;

	///@brief OpenCV KNN parameters
	typedef struct TrainNodeCvKNNParams {
		size_t 	maxSamples;					///< Maximum number of samples to be used in training. 0 means using all the samples

		TrainNodeCvKNNParams() {}
		TrainNodeCvKNNParams(int _maxSamples) : maxSamples(_maxSamples) {}
	} TrainNodeCvKNNParams;

	const TrainNodeCvKNNParams TRAIN_NODE_CV_KNN_PARAMS_DEFAULT = TrainNodeCvKNNParams(0);

	// ====================== OpenCV k-Nearest Neighbors Train Class =====================
	/**
	* @ingroup moduleTrainNode
	* @brief OpenCV Nearest Neighbor training class
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

	
	private:
		void		  init(TrainNodeCvKNNParams params);		// This function is called by both constructors

	
	protected:
		Ptr<ml::KNearest>			  m_pKNN;					///< k-NearestNeighbors
		CSamplesAccumulator			* m_pSamplesAcc;			///< Samples Accumulator

	};
}