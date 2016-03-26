// Contrast-Sensitive Potts training model for pairwise potentials
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "TrainEdgePotts.h"

namespace DirectGraphicalModels
{
	/**
	* @brief Penalization approach flag
	* @details These flags specify which penalization function \f$\mathcal{P}(d;\,\lambda)\f$ for penalizing the diagonal elements of the edge potential matrix will be used. 
	* Argument \f$d\f$ usually represent the Euclidean distance between two feature vectors, which correspond to the both nodes of an edge, and thus, is a measure of
	* similarity (or contrast) between them. Parameter \f$\lambda\f$ defines the penalization strength. The penaliztion functions are given as follows:
	* - \b Charbonnier \b penalization \b approach:   \f$\mathcal{P}(d;\,\lambda) = \frac{\lambda}{\sqrt{\lambda^2 + d^2}}\f$
	* - \b Perrona-Malik \b penalization \b approach: \f$\mathcal{P}(d;\,\lambda) = \frac{\lambda^2}{\lambda^2 + d^2}\f$
	* - \b Exponential \b penalization \b approach:   \f$\mathcal{P}(d;\,\lambda) = e^{-\lambda d^2}\f$
	*/
	enum ePotPenalApproach {
		eP_APP_PEN_CHAR,	///< Charbonnier penalization approach 
		eP_APP_PEN_PM,		///< Perrona-Malik penalization approach
		eP_APP_PEN_EXP		///< Exponential penalization approach
	}; 


	// ============================= Contrast-Sensitive Potts Train Class =============================
	/**
	* @ingroup moduleTrainEdge
	* @brief Contrast-Sensitive Potts training class
	* @details This class improves the @ref CTrainEdgePotts class by considering the \a contrast in test-data between edge's nodes.
	* In other respects this class is still training-data-independent, and thus may be applied without edge training procedure, \a i.e.
	* functions reset(), save(), load(), addFeatureVecs() are unnecessary here.
	* > This class may be used in case when the training-data for edges is not available.	
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainEdgePottsCS : public CTrainEdgePotts
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features	
		* @param penApproach Flag specifying the penalization approach for the edge potential matrix (Ref. @ref ePotPenalApproach)
		*/
		DllExport CTrainEdgePottsCS(byte nStates, byte nFeatures, ePotPenalApproach penApproach = eP_APP_PEN_EXP) 
			: CTrainEdgePotts(nStates, nFeatures)
			, CBaseRandomModel(nStates)
			, m_penApproach(penApproach) 
		{}
		DllExport virtual ~CTrainEdgePottsCS(void) {}


	protected:
		/**
		* @brief Returns the contrast-sensitive edge potentials
		* @details This function returns matrix with diagonal elements equal to parameter \f$\vec{\theta}\f$ provided through argument \b params 
		* and penalized with a penalization function \f$\mathcal{P}\f$, which depends on the difference between arguments \b featureVector1 and \b featureVector2; 
		* all the other elements are 1's: 
		* \f[edgePot[nStates][nStates] = \begin{bmatrix} \theta_1\mathcal{P} & 1 & \cdots & 1 \\ 1 & \theta_2\mathcal{P} & \cdots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \cdots & \theta_{nStates}\mathcal{P} \end{bmatrix}, \f]
		* where \f$\mathcal{P}\equiv\mathcal{P}(d;\,\lambda)\f$ is the penalization function, where \f$d = ||\textbf{f}_1 - \textbf{f}_2||_2\f$ is the Euclidean distance between feature vectors,  
		* and \f$\lambda\f$ is a parameter provided through argument \b params. For more details on penalization fuction, please refere to @ref ePotPenalApproach.
		* @param featureVector1 Multi-dimensinal point \f$\textbf{f}_1\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the first node of the edge
		* @param featureVector2 Multi-dimensinal point \f$\textbf{f}_2\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the second node of the edge
		* @param params Array of control parameters \f$\{\vec{\theta},\lambda\} \f$. \f$\vec{\theta}\f$ may consist either from \a one parameter (in this case all the diagonal elemets will be the same), 
		* or from \a nStates parameters, specifying smoothness strength for each state (class) individually; \f$\lambda\f$ consists from \a one parameter.
		* @param params_len The length of the \b params parameter. It must be equal to (\a 1 || \a nStates) + \a 1.
		* @return The edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		* > If \b featureVector1 or \b featureVector2 is empty, the function returns the test-data-independent Potts potential: @ref CTrainEdgePotts::calculateEdgePotentials()
		*/		
		DllExport virtual Mat	calculateEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, float *params, size_t params_len) const;


	private:
		ePotPenalApproach	m_penApproach;

	};
}
