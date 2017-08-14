// Artificial Neural Network (based on OpenCV) training class interface
// Written by Sergey G. Kosov in 2017 for Project X
#pragma once

#include "TrainNode.h"

namespace DirectGraphicalModels
{
	class CSamplesAccumulator;

	///@brief OpenCV SVM parameters
	typedef struct TrainNodeCvANNParams {
		word	numLayers;							///< Number of layers of neurons
		double	weightScale;						///<
		double	momentumScale;						///<
		int		maxCount;							///< Max number of trees in the forest (time / accuracy)
		double	epsilon;							///< Accuracy
		int		term_criteria_type;					///< Termination cirteria type (according the the two previous parameters)	
		size_t 	maxSamples;							///< Maximum number of samples to be used in training. 0 means using all the samples

		TrainNodeCvANNParams() {}
		TrainNodeCvANNParams(word _numLayers, double _weightScale, double _momentumScale, int _maxCount, double _epsilon, int _term_criteria_type, int _maxSamples) : numLayers(_numLayers), weightScale(_weightScale), maxCount(_maxCount), epsilon(_epsilon), term_criteria_type(_term_criteria_type), maxSamples(_maxSamples) {}
	} TrainNodeCvANNParams;

	const TrainNodeCvANNParams TRAIN_NODE_CV_ANN_PARAMS_DEFAULT = TrainNodeCvANNParams(
																							5,		// Num layers
																							0.0001,	// Backpropagation Weight Scale
																							0.0,	// Backpropagation Momentum Scale
																							100,	// Max number of trees in the forest (time / accuracy)
																							0.01,	// Forest accuracy
																							TermCriteria::MAX_ITER | TermCriteria::EPS, // Termination cirteria (according the the two previous parameters)
																							0		// Maximum number of samples to be used in training. 0 means using all the samples
																						);

	// ====================== OpenCV Support Vector Machines Train Class =====================
	/**
	* @ingroup moduleTrainNode
	* @brief OpenCV Support Vector Machines training class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeCvANN : public CTrainNode {
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params SVM parameters (Ref. @ref TrainNodeCvSVMParams)
		*/
		DllExport CTrainNodeCvANN(byte nStates, word nFeatures, TrainNodeCvANNParams params = TRAIN_NODE_CV_ANN_PARAMS_DEFAULT);
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxSamples Maximum number of samples to be used in training
		* > Default value \b 0 means using all the samples.<br>
		* > If another value is specified, the class for training will use \b maxSamples random samples from the whole amount of samples, added via addFeatureVec() function
		*/
		DllExport CTrainNodeCvANN(byte nStates, word nFeatures, size_t maxSamples);
		DllExport virtual ~CTrainNodeCvANN(void);

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
		void		  init(TrainNodeCvANNParams params);		// This function is called by both constructors


	protected:
		Ptr<ml::ANN_MLP>			  m_pANN;					///< Support Vector Machine
		CSamplesAccumulator			* m_pSamplesAcc;			///< Samples Accumulator

	};
}