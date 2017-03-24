// Nearest Neighbor training class interface
// Written by Sergey G. Kosov in 2017 for Project X
#pragma once

#include "TrainNode.h"

namespace DirectGraphicalModels
{
	class CSamplesAccumulator;
	class CKDTree;
	
	// ====================== Nearest Neighbor Train Class =====================
	/**
	* @ingroup moduleTrainNode
	* @brief Nearest Neighbor training class
	* @details This class implements the <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm" target="blank">k-nearest neighbors classifier (k-NN)</a>,
	* where the input consists of the k closest training samples in the feature space and the output depends on k-Nearest Neighbors.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainNodeKNN : public CTrainNode
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		* @param maxSamples Maximum number of samples to be used in training.
		* > Default value \b 0 means using all the samples.<br>
		* > If another value is specified, the class for training will use \b maxSamples random samples from the whole amount of samples, added via addFeatureVec() function
		*/
		DllExport CTrainNodeKNN(byte nStates, word nFeatures, size_t maxSamples = 0);
		DllExport ~CTrainNodeKNN(void);

		DllExport virtual void	  reset(void);

		DllExport virtual void	  addFeatureVec(const Mat &featureVector, byte gt);
		DllExport virtual void	  train(bool doClean = false);


	protected:
		DllExport virtual void	  saveFile(FILE *pFile) const {}
		DllExport virtual void	  loadFile(FILE *pFile) {}
		/**
		* @brief Calculates the node potential, based on the feature vector.
		* @details This function calculates the potentials of the node, described with the sample \b featureVector (\f$ \textbf{f} \f$):
		* \f[ nodePot_s = prior_s\cdot\prod_{f\in\mathbb{F}} (H_{s,f}.data[\textbf{f}_f] / H_{s,f}.n); \forall s\in\mathbb{S}, \f]
		* where \f$\mathbb{S}\f$ and \f$\mathbb{F}\f$ are sets of all states (classes) and features correspondently. In other words, the indexes:
		* \f$ s \in [0; nStates) \f$ and \f$ f \in [0; nFeatures) \f$.
		* Here \f$ H.data[256] \f$ is a 1D histogram, \f$ H.n \f$ is the number of entries in histogram, \a i.e.  \f$ H.n = \sum^{255}_{i = 0} H.data[i] \f$.
		* And \f$ \textbf{f}_f \in [0; 255], \forall f \in [0; nFeatures) \f$, \a i.e. has (type: CV_8UC1).
		* @param[in]	featureVector Multi-dimensinal point \f$\textbf{f}\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1)
		* @param[in,out]	potential %Node potentials: Mat(size: nStates x 1; type: CV_32FC1). This parameter should be preinitialized and set to value 0.
		* @param[in,out]	mask Relevant %Node potentials: Mat(size: nStates x 1; type: CV_8UC1). This parameter should be preinitialized and set to value 1 (all potentials are relevant).
		*/
		DllExport virtual void calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const;


	protected:
		CSamplesAccumulator * m_pSamplesAcc;
		CKDTree				* m_pTree;
	
	private:

	};
}