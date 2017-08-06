// Random Forest (based on OpenCV) training class interface
// Written by Sergey G. Kosov in 2012 for Project X
#pragma once

#include "TrainNode.h"

namespace DirectGraphicalModels
{
	class ml::RTrees;
	class CSamplesAccumulator;

	/// @brief OpenCV Random Forest parameters
	typedef struct TrainNodeCvRFParams {
		int		max_depth;							///< Max depth
		int		min_sample_count;					///< Min sample count (1% of all data)
		float	regression_accuracy;				///< Regression accuracy (0 means N/A here)
		bool	use_surrogates;						///< Compute surrogate split, no missing data
		int		max_categories;						///< Max number of categories (use sub-optimal algorithm for larger numbers) 
		bool	calc_var_importance;				///< Calculate variable importance (must be \a true in order to use CTrainNodeCvRF::getFeatureImportance function)
		int		nactive_vars;						///< Number of variables randomly selected at node and used to find the best split(s). (0 means the \f$ \sqrt{nFeatures} \f$)
		int		maxCount;							///< Max number of trees in the forest (time / accuracy)
		double	epsilon;							///< Forest accuracy
		int		term_criteria_type;					///< Termination cirteria type (according the the two previous parameters)
		size_t 	maxSamples;							///< Maximum number of samples to be used in training. 0 means using all the samples

		TrainNodeCvRFParams() {}
		TrainNodeCvRFParams(int _max_depth, int _min_sample_count, float _regression_accuracy, bool _use_surrogates, int _max_categories, bool _calc_var_importance, int _nactive_vars,	int _maxCount, double _epsilon, int _term_criteria_type, size_t _maxSamples) : max_depth(_max_depth), min_sample_count(_min_sample_count), regression_accuracy(_regression_accuracy), use_surrogates(_use_surrogates), max_categories(_max_categories), calc_var_importance(_calc_var_importance), nactive_vars(_nactive_vars), maxCount(_maxCount), epsilon(_epsilon), term_criteria_type(_term_criteria_type), maxSamples(_maxSamples) {}
	} TrainNodeCvRFParams;

	const TrainNodeCvRFParams TRAIN_NODE_CV_RF_PARAMS_DEFAULT = TrainNodeCvRFParams(
																25,		// Max depth
																5,		// Min sample count (1% of all data)
																0,		// Regression accuracy (0 means N/A here)
															    false,	// Compute surrogate split, no missing data
																15,		// Max number of categories (use sub-optimal algorithm for larger numbers) 
																false,	// Calculate variable importance 
																4,		// Number of variables randomly selected at node and used to find the best split(s). 0 means sqrt(nFeatures) 
																100,	// Max number of trees in the forest (time / accuracy)
																0.01,	// Forest accuracy
																TermCriteria::MAX_ITER | TermCriteria::EPS, // Termination cirteria (according the the two previous parameters)
																0		// Maximum number of samples to be used in training. 0 means using all the samples
																);

	// =========================== OpenCV RF Train Class ===========================
	/**
	* @ingroup moduleTrainNode
	* @brief OpenCV Random Forest training class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeCvRF : public CTrainNode
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params Random Forest parameters (Ref. @ref TrainNodeCvRFParams)
		*/
		DllExport CTrainNodeCvRF(byte nStates, word nFeatures, TrainNodeCvRFParams params = TRAIN_NODE_CV_RF_PARAMS_DEFAULT);
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxSamples Maximum number of samples to be used in training. 
		* > Default value \b 0 means using all the samples.<br>
		* > If another value is specified, the class for training will use \b maxSamples random samples from the whole amount of samples, added via addFeatureVec() function
		*/
		DllExport CTrainNodeCvRF(byte nStates, word nFeatures, size_t maxSamples);
		DllExport ~CTrainNodeCvRF(void);

		DllExport void	reset(void);		
		DllExport void	save(const std::string &path, const std::string &name = std::string(), short idx = -1) const; 
		DllExport void	load(const std::string &path, const std::string &name = std::string(), short idx = -1); 
	
		DllExport void	addFeatureVec(const Mat &featureVector, byte gt);
		DllExport void	train(bool doClean = false);

		/**
		* @brief Returns the feature importance vector
		* @details The method returns the feature importance vector, computed at the training stage when TrainNodeCvRFParams::calc_var_importance is set to true. 
		* @retval NULL : Empty Mat() on error (TrainNodeCvRFParams::calc_var_importance flag is not set)
		* @retval feature_importance : Mat(size: 1 x nFeatures; type: CV_32FC1)
		*/
		DllExport Mat	getFeatureImportance(void) const; 

	
	protected:
		DllExport void	saveFile(FILE *pFile) const { }
		DllExport void	loadFile(FILE *pFile) { }
		DllExport void	calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const;


	protected:
		Ptr<ml::RTrees>			  m_pRF;						///< Random Forest
		CSamplesAccumulator	* m_pSamplesAcc;				///< Samples Accumulator


	private:
		void			init(TrainNodeCvRFParams params);	// This function is called by both constructors
		

	private:
		TrainNodeCvRFParams	  m_params;
	};
}

