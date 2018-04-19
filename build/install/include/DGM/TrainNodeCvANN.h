// Artificial Neural Network (based on OpenCV) training class interface
// Written by Sergey G. Kosov in 2017 for Project X
#pragma once

#include "TrainNode.h"

namespace DirectGraphicalModels
{
	class CSamplesAccumulator;

	///@brief OpenCV  Artificial neural network parameters
	typedef struct TrainNodeCvANNParams {
		word	numLayers;							///< Number of layers of neurons
		double	weightScale;						///< Strength of the weight gradient term. The recommended value is about 0.1. Default value is 0.1.
		double	momentumScale;						///< Strength of the momentum term (the difference between weights on the 2 previous iterations). This parameter provides some inertia to smooth the random fluctuations of the weights. It can vary from 0 (the feature is disabled) to 1 and beyond. The value 0.1 or so is good enough. Default value is 0.1.
		int		maxCount;							///< The maximum number of iterations (time / accuracy)
		double	epsilon;							///< The desired accuracy or change in parameters at which the iterative algorithm stops 
		int		term_criteria_type;					///< Termination cirteria type (according the the two previous parameters)	
		size_t 	maxSamples;							///< Maximum number of samples to be used in training. 0 means using all the samples

		TrainNodeCvANNParams() {}
		TrainNodeCvANNParams(word _numLayers, double _weightScale, double _momentumScale, int _maxCount, double _epsilon, int _term_criteria_type, int _maxSamples) : numLayers(_numLayers), weightScale(_weightScale), maxCount(_maxCount), epsilon(_epsilon), term_criteria_type(_term_criteria_type), maxSamples(_maxSamples) {}
	} TrainNodeCvANNParams;

	const TrainNodeCvANNParams TRAIN_NODE_CV_ANN_PARAMS_DEFAULT =	TrainNodeCvANNParams(
																	5,		// Num layers
																	0.0001,	// Backpropagation Weight Scale
																	0.1,	// Backpropagation Momentum Scale
																	100,	// The maximum number of iterations (time / accuracy)
																	0.01,	// The desired accuracy or change in parameters at which the iterative algorithm stops 
																	TermCriteria::MAX_ITER | TermCriteria::EPS, // Termination cirteria (according the the two previous parameters)
																	0		// Maximum number of samples to be used in training. 0 means using all the samples
																	);

	// ====================== OpenCV Artificial Neural Network Train Class =====================
	/**
	* @ingroup moduleTrainNode
	* @brief OpenCV Artificial neural network training class
	* @details This class implements the <a href="https://en.wikipedia.org/wiki/Artificial_neural_network" target="blank">artificial neural network classifier (ANN)</a>.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeCvANN : public CTrainNode {
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params ANN parameters (Ref. @ref TrainNodeCvANNParams)
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
		Ptr<ml::ANN_MLP>			  m_pANN;					///< Artificial Neural Network 
		CSamplesAccumulator			* m_pSamplesAcc;			///< Samples Accumulator

	};
}