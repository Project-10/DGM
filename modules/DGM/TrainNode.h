// Base abstract class for random model nodes training
// Written by Sergey G. Kosov in 2013 for Project X
#pragma once

#include "ITrain.h"

namespace DirectGraphicalModels
{
	/// Types of the node potential function 
	enum NodeRandomModel : byte { 
		Bayes = 0,				///< Bayes Model
		GMM, 					///< Gaussian Mixture Model
		CvGMM, 					///< OpenCV Gaussian Mixture Model
		KNN, 					///< Nearest Neighbor
		CvKNN, 					///< OpenCV Nearest Neighbor
		CvRF, 					///< OpenCV Random Forest
		MsRF, 					///< MicroSoft Random Forest
		CvANN, 					///< OpenCV Artificial Neural Network
		CvSVM, 					///< OpenCV Support Vector Machines

		GM, 					///< Gaussian Model
		CvGM, 					///< OpenCV Gaussian Model
	 };

	// ============================= Node Train Class =============================
	/**
	* @ingroup moduleTrainNode
	* @brief Base abstract class for node potentials training
	* @details The common usage for node potentials training is as follow:
	* @code
	* CTrainNode *t = new CTrainNodeXXX(nStates, nFeatures);
	*
	* // "adding data" phase 
	* for (int i = 0; i < NUMBER_OF_TRAINING_DATA; i++) t->addFeatureVec(train_featurVector[i], groundtruth_class[i]);
	*
	* // training
	* t->train();
	*
	* // "getting data" phase
	* for (int i = 0; i < NUMBER_OF_TEST_DATA; i++) predicted_class[i] = t->getNodePotentials(test_featurVector[i]);
	* 
	* delete t;
	* @endcode
	* See @ref demotrain for more details
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNode : public ITrain
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		*/
		DllExport CTrainNode(byte nStates, word nFeatures)
			    : CBaseRandomModel(nStates)
				, ITrain(nStates, nFeatures)
				, m_mask(nStates, 1, CV_8UC1)
		{}
		DllExport virtual ~CTrainNode(void) = default;
	
		/**
		* @brief Factory method returning node trainer object 
		* @note The resulting node trainer object is created with default parameters
		* @param nodeRandomModel Type of desired random model (Ref. @ref NodeRandomModel)
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @return Tne pointer to the concrete implementation of the node trainer class
		*/
		DllExport static std::shared_ptr<CTrainNode> create(byte nodeRandomModel, byte nStates, word nFeatures);
		/**
		* @brief Adds a block of new feature vectors
		* @details Used to add multiple \b featureVectors, corresponding to the ground-truth states (classes) \b gt for training
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC(nFeatures))
		* @param gt Matrix, each element of which is a ground-truth state (class)
		*/			
		DllExport void			addFeatureVecs(const Mat &featureVectors, const Mat &gt);
		/**
		* @brief Adds a block of new feature vectors
		* @details Used to add multiple \b featureVectors, corresponding to the ground-truth states (classes) \b gt for training
		* @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
		* @param gt Matrix, each element of which is a ground-truth state (class)
		*/
		DllExport void			addFeatureVecs(const vec_mat_t &featureVectors, const Mat &gt);
		/**
		* @brief Adds new feature vector
		* @details Used to add a \b featureVector, corresponding to the ground-truth state (class) \b gt for training
		* @param featureVector Multi-dimensinal point: Mat(size: nFeatures x 1; type: CV_8UC1)
		* @param gt Corresponding ground-truth state (class)
		*/		
		DllExport virtual void	addFeatureVec(const Mat &featureVector, byte gt) = 0;	
		DllExport virtual void	train(bool doClean = false) {}
		/**
		* @brief Returns a block of node potentials, based on the block of feature vector
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC(nFeatures))
		* @param weights The block of weighting parameters Mat(type: CV_32FC1). If empty, values 1 are used.
		* @param Z The value of <a href="https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)">partition function</a>.
		* In order to convert potential to the probability, it is multiplied by \f$1/Z\f$.
		* If \f$Z\leq0\f$, the resulting node potentials are normalized to 100, independently for each potential.
		*/
		DllExport Mat			getNodePotentials(const Mat &featureVectors, const Mat &weights = Mat(), float Z = 0.0f) const;
		/**
		* @brief Returns a block of node potentials, based on the block of feature vector
		* @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
		* @param weights The block of weighting parameter Mat(type: CV_32FC1)
		* @param Z The value of <a href="https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)">partition function</a>.
		* In order to convert potential to the probability, it is multiplied by \f$1/Z\f$.
		* If \f$Z\leq0\f$, the resulting node potentials are normalized to 100, independently for each potential.
		*/
		DllExport Mat			getNodePotentials(const vec_mat_t &featureVectors, const Mat &weights = Mat(), float Z = 0.0f) const;
		/**
		* @brief Returns the node potential, based on the feature vector
		* @details This function calls calculateNodePotentials() function, which should be implemented in derived classes. After that,
		* the resulting node potential is powered by parameter \b weight.
		* @param featureVector Multi-dimensinal point \f$\textbf{f}\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1)
		* @param weight The weighting parameter (default value is 1)
		* @param Z The value of <a href="https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)">partition function</a>.
		* In order to convert potential to the probability, it is multiplied by \f$1/Z\f$. 
		* If \f$Z\leq0\f$, the resulting node potentials are normalized to 100, independently for every function call. 
		* @return Normalized %node potentials on success: Mat(size: nStates x 1; type: CV_32FC1); 
		*/		
		DllExport Mat			getNodePotentials(const Mat &featureVector, float weight, float Z = 0.0f) const;


	protected:
		/**
		* @brief Calculates the node potential, based on the feature vector
		* @details This function calculates the potentials of the node, described with the sample \b featureVector, being in each state 
		* (belonging to each class). These potentials are united in the node potential vector: 
		* \f[nodePot[nStates] = f(\textbf{f}[nFeatures]).\f] Functions \f$ f \f$ must be implemented in derived classes.
		* @param[in]	featureVector Multi-dimensinal point \f$\textbf{f}\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1)
		* @param[in,out]	potential %Node potentials: Mat(size: nStates x 1; type: CV_32FC1). This parameter should be preinitialized and set to value 0.
		* @param[in,out]	mask Relevant %Node potentials: Mat(size: nStates x 1; type: CV_8UC1). This parameter should be preinitialized and set to value 1 (all potentials are relevant).
		*/		
		DllExport virtual void calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const = 0;
		

	private:
		Mat	m_mask;
	};
}

