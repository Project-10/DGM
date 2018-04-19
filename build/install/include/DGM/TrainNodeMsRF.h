// Random Forest (based on Microsof Sherwood library) training class interface
// Written by Sergey G. Kosov in 2013 for Project X
#pragma once

#include "TrainNode.h"

//#ifdef USE_SHERWOOD

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood {
	class LinearFeatureResponse;
	class HistogramAggregator;
	class DataPointCollection;
	template<class F, class S> class Forest;
	struct TrainingParameters;
}}}

namespace sw = MicrosoftResearch::Cambridge::Sherwood;

namespace DirectGraphicalModels
{
	class CSamplesAccumulator;
	
	///@brief Microsoft Research Random Forest parameters
	typedef struct TrainNodeMsRFParams {
		int				max_decision_levels;						///< Maximum number of the decision levels
		int				num_of_candidate_features;					///< Number of candidate features
		unsigned int	num_of_candidate_thresholds_per_feature;	///< Number of candidate thresholds (per feature)
		int				num_ot_trees;								///< Number of trees in the forest (time / accuracy)
		bool			verbose;									///< Verbose mode
		size_t			maxSamples;									///< Maximum number of samples to be used in training. 0 means using all the samples

		TrainNodeMsRFParams() {}
		TrainNodeMsRFParams(int _max_decision_levels, int _num_of_candidate_features, unsigned int _num_of_candidate_thresholds_per_feature, int _num_ot_trees, bool _verbose, int _maxSamples) : max_decision_levels(_max_decision_levels), num_of_candidate_features(_num_of_candidate_features), num_of_candidate_thresholds_per_feature(_num_of_candidate_thresholds_per_feature), num_ot_trees(_num_ot_trees), verbose(_verbose), maxSamples(_maxSamples) {}
	} TrainNodeMsRFParams;

	const TrainNodeMsRFParams TRAIN_NODE_MS_RF_PARAMS_DEFAULT = TrainNodeMsRFParams(
																10,		// Maximum number of the decision levels
																10,		// Number of candidate features
																10,		// Number of candidate thresholds (per feature)
																10,		// Number of trees in the forest (time / accuracy)
																false,	// Verbose mode
																0		// Maximum number of samples to be used in training. 0 means using all the samples
																);
	
	// =========================== Microsoft RF Train Class ===========================
	/**
	* @ingroup moduleTrainNode
	* @brief Microsoft Sherwood Random Forest training class
	* @details This class is based on the <a href="http://research.microsoft.com/en-us/downloads/52d5b9c3-a638-42a1-94a5-d549e2251728/">Sherwood C++ code library for decision forests</a> v.1.0.0
	* > In order to use the Sherwood library, DGM must be built with the \b USE_SHERWOOD flag
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeMsRF : public CTrainNode
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param params Random Forest parameters (Ref. @ref TrainNodeMsRFParams)
		*/
		DllExport CTrainNodeMsRF(byte nStates, word nFeatures, TrainNodeMsRFParams params = TRAIN_NODE_MS_RF_PARAMS_DEFAULT);
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxSamples Maximum number of samples to be used in training.
		* > Default value \b 0 means using all the samples.<br>
		* > If another value is specified, the class for training will use \b maxSamples random samples from the whole amount of samples, added via addFeatureVec() function		
		* @note This implementation of the random forest is not weighted
		*/
		DllExport CTrainNodeMsRF(byte nStates, word nFeatures, size_t maxSamples);
		DllExport virtual ~CTrainNodeMsRF(void);

		/**
		* @brief Resets class variables
		* @details Allows to re-use the class
		* @note This function may be extremely slow
		* @todo Check! It may be very slow here
		*/
		DllExport void	reset(void);
		DllExport void	save(const std::string &path, const std::string &name = std::string(), short idx = -1) const; 
		DllExport void  load(const std::string &path, const std::string &name = std::string(), short idx = -1); 

		DllExport void	addFeatureVec(const Mat &featureVector, byte gt);
		DllExport void	train(bool doClean = false);


	protected:
		DllExport void saveFile(FILE *pFile) const { }
		DllExport void loadFile(FILE *pFile) { }
		DllExport void calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const;


	protected:
		std::auto_ptr<sw::Forest<sw::LinearFeatureResponse, sw::HistogramAggregator>> 	 m_pRF;			///< Random Forest classifier
		CSamplesAccumulator															   * m_pSamplesAcc;	///< Samples Accumulator


	private:
		void		  init(TrainNodeMsRFParams params);													// This function is called by both constructors


	private:
		std::auto_ptr<sw::TrainingParameters>											m_pParams;
	};
}
//#endif
