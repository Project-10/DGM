// Potts training model for pairwise potentials
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "TrainEdge.h"

namespace DirectGraphicalModels
{
	// ============================= Potts Train Class =============================
	/**
	* @ingroup moduleTrainEdge
	* @brief Potts edge training class
	* @details This class implements the <a href="http://en.wikipedia.org/wiki/Potts_model">Potts model</a> for edge (pairwise) potentials,
	* which is both training- and test-data-independent, and thus may be applied without edge training procedure, \a i.e.
	* functions reset(), save(), load(), addFeatureVecs() are unnecessary here.
	* > This class may be used in case when the training- and test-data for edges is not available.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainEdgePotts : public CTrainEdge
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features
		*/		
		DllExport CTrainEdgePotts(byte nStates, word nFeatures) : CTrainEdge(nStates, nFeatures), CBaseRandomModel(nStates) {}
		DllExport virtual ~CTrainEdgePotts(void) {}

		DllExport virtual void	reset(void) {}	

		DllExport virtual void	addFeatureVecs(const Mat &featureVector1, byte gt1, const Mat &featureVector2, byte gt2) {}
		/**
		* @brief Returns the data-independent edge potentials
		* @details This function returns matrix with diagonal elements equal to the argument \b val, all the other elements are 1's, what imitates the Potts model. 
		* \f[edgePot[nStates][nStates] = \begin{bmatrix} val & 1 & \cdots & 1 \\ 1 & val & \cdots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \cdots & val \end{bmatrix} \f]
		* @param val The diagonal element of the matrix
		* @param nStates Number of states (classes)
		* @return The edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)	
		*/
		DllExport static Mat	getEdgePotentials(float val, byte nStates);		
		/**
		* @brief Returns the data-independent edge potentials
		* @details This function returns matrix with diagonal elements specified by argument array \b values, all the other elements are 1's, what imitates the Potts model. 
		* \f[edgePot[nStates][nStates] = \begin{bmatrix} values_1 & 1 & \cdots & 1 \\ 1 & values_2 & \cdots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \cdots & values_{nStates} \end{bmatrix} \f]
		* @param values The array of diagonal elements of the matrix. In must include \a nStates values, specifying smoothness strength for each state (class) individually.
		* @param nStates Number of states (classes)
		* @return The edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)	
		*/
		DllExport static Mat	getEdgePotentials(float *values, byte nStates);


	protected:
		DllExport virtual void  saveFile(FILE *pFile) const {}
		DllExport virtual void  loadFile(FILE *pFile) {} 
		/**
		* @brief Returns the data-independent edge potentials
		* @details This function returns matrix with diagonal elements equal to parameter \f$\vec{\theta}\f$ provided through argument \b params; 
		* all the other elements are 1's - this imitates the Potts model. 
		* \f[edgePot[nStates][nStates] = \begin{bmatrix} \theta_1 & 1 & \cdots & 1 \\ 1 & \theta_2 & \cdots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \cdots & \theta_{nStates} \end{bmatrix} \f]
		* @param featureVector1 Multi-dimensinal point \f$\textbf{f}_1\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the first node of the edge
		* > It is not used in the Potts model, thus may be empty Mat()
		* @param featureVector2 Multi-dimensinal point \f$\textbf{f}_2\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the second node of the edge
		* > It is not used in the Potts model, thus may be empty Mat()
		* @param params Array of control parameters \f$\vec{\theta}\f$, which may consist either from \a one parameter (in this case all the diagonal elemets will be the same), 
		* or from \a nStates parameters, specifying smoothness strength for each state (class) individually.
		* @param params_len The length of the \b params parameter. It must be equal to (\a 1 || \a nStates).
		* @return The edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual Mat	calculateEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, float *params, size_t params_len) const;
	};
}
