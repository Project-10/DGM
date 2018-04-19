// Support Vector Machines (based on OpenCV) training class interface
// Written by Sergey G. Kosov in 2017 for Project X
#pragma once

#include "TrainNode.h"

namespace DirectGraphicalModels
{
	class CSamplesAccumulator;

	///@brief OpenCV Support Vector machine parameters
	typedef struct TrainNodeCvSVMParams {
		double  C;							///< Parameter C of a SVM optimization problem
		int		maxCount;					///< The maximum number of iterations (time / accuracy)
		double	epsilon;					///< The desired accuracy or change in parameters at which the iterative algorithm stops 
		int		term_criteria_type;			///< Termination cirteria type (according the the two previous parameters)
		size_t 	maxSamples;					///< Maximum number of samples to be used in training. 0 means using all the samples

		TrainNodeCvSVMParams() {}
		TrainNodeCvSVMParams(double C, int maxCount, double epsilon, int term_criteria_type, int maxSamples) : C(C), maxCount(maxCount), epsilon(epsilon), term_criteria_type(term_criteria_type), maxSamples(maxSamples) {}
	} TrainNodeCvSVMParams;

	const TrainNodeCvSVMParams TRAIN_NODE_CV_SVM_PARAMS_DEFAULT =	TrainNodeCvSVMParams(
																	0.4,	// Parameter C of a SVM optimization problem
																	10000,	// The maximum number of iterations (time / accuracy)
																	0.01,	// The desired accuracy or change in parameters at which the iterative algorithm stops 
																	TermCriteria::MAX_ITER | TermCriteria::EPS, // Termination cirteria (according the the two previous parameters)
																	0		// Maximum number of samples to be used in training. 0 means using all the samples
																	);

	// ====================== OpenCV Support Vector Machines Train Class =====================
	/**
	* @ingroup moduleTrainNode
	* @brief OpenCV Support Vector Machines training class
	* @details This class implements the <a href="https://en.wikipedia.org/wiki/Support_vector_machine" target="blank">Support vector machine classifier (SVM)</a>.
	* @note This trainer was not well-tested.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeCvSVM : public CTrainNode {
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params SVM parameters (Ref. @ref TrainNodeCvSVMParams)
		*/
		DllExport CTrainNodeCvSVM(byte nStates, word nFeatures, TrainNodeCvSVMParams params = TRAIN_NODE_CV_SVM_PARAMS_DEFAULT);
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxSamples Maximum number of samples to be used in training
		* > Default value \b 0 means using all the samples.<br>
		* > If another value is specified, the class for training will use \b maxSamples random samples from the whole amount of samples, added via addFeatureVec() function
		*/
		DllExport CTrainNodeCvSVM(byte nStates, word nFeatures, size_t maxSamples);
		DllExport virtual ~CTrainNodeCvSVM(void);

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
		void		  init(TrainNodeCvSVMParams params);		// This function is called by both constructors


	protected:
		Ptr<ml::SVM>				  m_pSVM;					///< Support Vector Machine
		CSamplesAccumulator			* m_pSamplesAcc;			///< Samples Accumulator

	};
}