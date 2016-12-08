// Base abstract class for random model edges training
// Written by Sergey G. Kosov in 2013-2015 for Project X
#pragma once

#include "ITrain.h"

namespace DirectGraphicalModels
{
	class CGraphExt;
	
	// ============================= Edge Train Class =============================
	/**
	* @ingroup moduleTrainEdge
	* @brief Base abstract class for edge potentials training
	* @details Refer to the @ref demotrain for the application and common usage example
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainEdge : public ITrain
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		*/
		DllExport CTrainEdge(byte nStates, word nFeatures) : ITrain(nStates, nFeatures), CBaseRandomModel(nStates) {}
		DllExport virtual ~CTrainEdge(void) {}

		/**
		* @brief Adds a block of new feature vectors
		* @details This function may be used only for basic graphical models, built with the CGraphExt::build() method. It extracts 
		* pairs of feature vectors with corresponding ground-truth values from blocks \b featureVectors and \b gt, according to the graph structure, 
		* provided via \b pGraph
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
		* @param gt Matrix, each element of which is a ground-truth state (class)
		* @param pGraph Pointer to the extended graph, which defines the structure of the graph
		*/
		DllExport void			addFeatureVecs(const Mat &featureVectors, const Mat &gt, const CGraphExt *pGraph);
		/**
		* @brief Adds a block of new feature vectors
		* @details This function may be used only for basic graphical models, built with the CGraphExt::build() method. It extracts
		* pairs of feature vectors with corresponding ground-truth values from blocks \b featureVectors and \b gt, according to the graph structure,
		* provided via \b pGraph
		* @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
		* @param gt Matrix, each element of which is a ground-truth state (class)
		* @param pGraph Pointer to the extended graph, which defines the structure of the graph
		*/
		DllExport void			addFeatureVecs(const vec_mat_t &featureVectors, const Mat &gt, const CGraphExt *pGraph);
		/**
		* @brief Adds a pair of feature vectors
		* @details Used to add \b featureVector1 and \b featureVector2, corresponding to the ground-truth states (classes) \b gt1 and \b gt2 for training.
		* Here the couple \b {featureVector1, \b gt1} corresponds to the first node of the edge, and the couple \b {featureVector2, \b gt2} - to the second node.
		* @param featureVector1 Multi-dimensinal point: Mat(size: nFeatures x 1; type: CV_8UC1), corresponding to the first node of the edge.
		* @param gt1 The ground-truth state (class) of the first node of the edge, given by \b featureVector1 
		* @param featureVector2 Multi-dimensinal point: Mat(size: nFeatures x 1; type: CV_8UC1), corresponding to the second node of the edge.
		* @param gt2 The ground-truth state (class) of the second node of the edge, given by \b featureVector2 
		*/		
		DllExport virtual void	addFeatureVecs(const Mat &featureVector1, byte gt1, const Mat &featureVector2, byte gt2) = 0;			
		DllExport virtual void	train(bool doClean = false) {}
		/**
		* @brief Returns the edge potential, based on the feature vectors
		* @details This function calls calculateEdgePotentials() function, which should be implemented in derived classes. After that,
		* the resulting edge potential is powered by parameter \b weight.
		* @param featureVector1 Multi-dimensinal point \f$\textbf{f}_1\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the first node of the edge
		* @param featureVector2 Multi-dimensinal point \f$\textbf{f}_2\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the second node of the edge
		* @param params Array of control parameters. Please refer to the concrete model implementation of the calculateEdgePotentials() function for more details
		* @param params_len The length of the \a params parameter
		* @param weight The weighting parameter
		* @return %Edge potentials on success: Mat(size: nStates x nStates; type: CV_32FC1)
		*/	
		DllExport Mat			getEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, float *params, size_t params_len, float weight = 1.0f) const; 


	protected:
		/**
		* @brief Calculates the edge potential, based on the feature vectors
		* @details This function calculates the potentials of an edge, described with two samples \b featureVector1 and \b featureVector2, correspondig to the both nodes defining that edge.
		* The resulting potentials of the two nodes being in each possible state (belonging to each possible class) at the same time, are united in the edge potential matrix: 
		* \f[edgePot[nStates][nStates] = f(\textbf{f}_1[nFeatures],\textbf{f}_2[nFeatures]).\f] 
		* Functions \f$ f \f$ must be implemented in derived classes.
		* @param featureVector1 Multi-dimensinal point \f$\textbf{f}_1\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the first node of the edge
		* @param featureVector2 Multi-dimensinal point \f$\textbf{f}_2\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the second node of the edge
		* @param params Array of control parameters. Please refere to the concrete model implementation of the calculateEdgePotentials() function for more details
		* @param params_len The length of the \b params parameter
		* @returns The edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/	
		DllExport virtual Mat	calculateEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, float *params, size_t params_len) const = 0;

	};
}
