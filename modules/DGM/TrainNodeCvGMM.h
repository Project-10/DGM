// Gaussian Mixture Model (based on OpenCV) training class interface
// Written by Sergey G. Kosov in 2012 for Project X
#pragma once

#include "TrainNode.h"

namespace DirectGraphicalModels
{
	///@brief OpenCV Random Forest parameters
	typedef struct TrainNodeCvGMMParams {
		word	numGausses;					///< The number of Gauss functions for approximation
		int		covariance_matrix_type;		///< Type of the covariance matrix
		int		maxCount;					///< Max number of iterations
		double	epsilon;					///< GMM accuracy
		int		term_criteria_type;			///< Termination cirteria type (according the the two previous parameters)
		int 	maxSamples;					///< Maximum number of samples to be used in training. 0 means using all the samples

		TrainNodeCvGMMParams() {}
		TrainNodeCvGMMParams(word _numGausses, int _covariance_matrix_type, int _maxCount, double _epsilon, int _term_criteria_type, int _maxSamples) : numGausses(_numGausses), covariance_matrix_type(_covariance_matrix_type), maxCount(_maxCount), epsilon(_epsilon), term_criteria_type(_term_criteria_type), maxSamples(_maxSamples) {}
	} TrainNodeCvGMMParams;

	const TrainNodeCvGMMParams TRAIN_NODE_CV_GMM_PARAMS_DEFAULT = TrainNodeCvGMMParams(
																16,											// Number of Gaussians
																ml::EM::COV_MAT_DIAGONAL,					// Covariance matrix type
																100,										// Max number of iterations
																0.01,										// GMM accuracy
																TermCriteria::MAX_ITER | TermCriteria::EPS,	// Termination cirteria (according the the two previous parameters)
																0											// Maximum number of samples to be used in training. 0 means using all the samples														
																);

	// =========================== OpenCV GMM Train Class ===========================
	/**
	* @ingroup moduleTrainNode
	* @brief OpenCV Gaussian Mixture Model training class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeCvGMM : public CTrainNode
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params Expectation Maximization parameters (Ref. @ref TrainNodeCvGMMParams)
		*/
		DllExport CTrainNodeCvGMM(byte nStates, word nFeatures, TrainNodeCvGMMParams params = TRAIN_NODE_CV_GMM_PARAMS_DEFAULT);
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxSamples Maximum number of samples to be used in training 
		* @param nGausses The number of mixture components in the Gaussian Mixture Model per state (class)
		*/
		DllExport CTrainNodeCvGMM(byte nStates, word nFeatures, int maxSamples, byte nGausses = TRAIN_NODE_CV_GMM_PARAMS_DEFAULT.numGausses);
		DllExport virtual ~CTrainNodeCvGMM(void);

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
		void		  init(TrainNodeCvGMMParams params);		// This function is called by both constructors

	private:
		static const double MIN_COEFFICIENT_BASE;

	private:
		std::vector<Ptr<ml::EM>>	m_vpEM;						// Expectation Maximization for GMM parameters estimation
		long double					m_minCoefficient;			// = 1;						// auxilary coefficient for scaling gaussian coefficients
		vec_mat_t					m_vSamplesAcc;				// = vec_mat_t(nStates);	// Samples container for all states
		vec_int_t					m_vNumInputSamples;			// = vec_int_t(nStates, 0);	// Amount of input samples for all states
		int							m_maxSamples;				// = INFINITY;				// for optimisation purposes
	};
}

