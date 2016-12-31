// Base abstract class for random model links (inter-layer edges) training
// Written by Sergey G. Kosov in 2016 for Project X
#pragma once

#include "ITrain.h"

namespace DirectGraphicalModels
{
	// ============================= Link Train Class =============================
	/**
	* @ingroup moduleTrainEdge
	* @brief Base abstract class for link (inter-layer edge) potentials training
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainLink : public ITrain
	{
	public:
		/**
		* @brief Constructor
		* @param nStatesBase Number of states (classes) for the base layer of the graphical model
		* @param nStatesOccl Number of states (classes) for the occlusion layer of the graphical model
		* @param nFeatures Number of features
		*/
		DllExport CTrainLink(byte nStatesBase, byte nStatesOccl, word nFeatures) 
			: ITrain(nStatesBase * nStatesOccl, nFeatures)
			, CBaseRandomModel(nStatesBase * nStatesOccl)
			, m_nStatesBase(nStatesBase)
			, m_nStatesOccl(nStatesOccl)
		{}
		DllExport virtual ~CTrainLink(void) {}

		
		/**
		* @brief Adds a block of new feature vectors
		* @details Used to add multiple \b featureVectors, corresponding to the ground-truth states (classes) \b gtb and \b gto for training
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
		* @param gtb Matrix, each element of which is a ground-truth state (class), corresponding to the base layer
		* @param gto Matrix, each element of which is a ground-truth state (class), corresponding to the occlusion layer
		*/
		DllExport void			addFeatureVec(const Mat &featureVectors, const Mat &gtb, const Mat &gto);
		/**
		* @brief Adds a block of new feature vectors
		* @details Used to add multiple \b featureVectors, corresponding to the ground-truth states (classes) \b gtb and \b gto for training
		* @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
		* @param gtb Matrix, each element of which is a ground-truth state (class), corresponding to the base layer
		* @param gto Matrix, each element of which is a ground-truth state (class), corresponding to the occlusion layer
		*/
		DllExport void			addFeatureVec(const vec_mat_t &featureVectors, const Mat &gtb, const Mat &gto);
		/**
		* @brief Adds a feature vector
		* @details Used to add \b featureVector, corresponding to the ground-truth states (classes) \b gtb and \b gto for training.
		* Here the couple \b {gtb, \b gto} corresponds to the nodes from base aod occlusion layers.
		* @param featureVector Multi-dimensinal point: Mat(size: nFeatures x 1; type: CV_8UC1), corresponding to both nodes of the link.
		* @param gtb The ground-truth state (class) of the first node of the edge, corresponding to the base layer
		* @param gto The ground-truth state (class) of the second node of the edge, corresponding to the occlusion layer
		*/
		DllExport virtual void	addFeatureVec(const Mat &featureVector, byte gtb, byte gto) = 0;
		DllExport virtual void	train(bool doClean = false) {}
		/**
		* @brief Returns the link potential, based on the feature vector
		* @details This function calls calculateLinkPotentials() function, which should be implemented in derived classes. After that,
		* the resulting edge potential is powered by parameter \b weight.
		* @param featureVector Multi-dimensinal point \f$\textbf{f}\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to both nodes of the link
		* @param weight The weighting parameter
		* @return %Edge potentials on success: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport Mat			getLinkPotentials(const Mat &featureVector, float weight = 1.0f) const;


	protected:
		/**
		* @brief Calculates the link potential, based on the feature vector
		* @details This function calculates the potentials of an edge, described with the sample \b featureVector, correspondig to the both nodes defining that edge.
		* The resulting potentials of the two nodes being in each possible state (one class from occlusion layer occlusion another class from the base layer), 
		* are united in the edge potential matrix:
		* \f[edgePot[nStates][nStates] = f(\textbf{f}[nFeatures])\f]
		* Functions \f$ f \f$ must be implemented in derived classes.
		* @param featureVector Multi-dimensinal point \f$\textbf{f}\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to both nodes of the link
		* @returns The edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual Mat	calculateLinkPotentials(const Mat &featureVector) const = 0;


	protected:
		byte m_nStatesBase;
		byte m_nStatesOccl;
	};
}