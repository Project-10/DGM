// Prior training model for pairwise potentials
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "TrainEdgePottsCS.h"
#include "PriorEdge.h"

namespace DirectGraphicalModels
{
	// ============================= Prior Edge Train Class =============================
	/**
	* @ingroup moduleTrainEdge
	* @brief Contrast-Sensitive Potts training with edge prior probability class
	* @details This class improves the @ref CTrainEdgePottsCS class by considering the \a prior probability of an edge to connect two nodes with concrete states (classes).
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrainEdgePrior : public CTrainEdgePottsCS, private CPriorEdge
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param nFeatures Number of features	
		* @param penApproach Flag specifying the penalization approach for the edge potential matrix (Ref. @ref ePotPenalApproach)
		* @param normApproach Flag specifying the co-occurance histogram matrix normalization approach (Ref. @ref ePotNormApproach)	
		*/
		DllExport CTrainEdgePrior(byte nStates, word nFeatures, ePotPenalApproach penApproach = eP_APP_PEN_EXP, ePotNormApproach normApproach = eP_APP_NORM_SYMMETRIC);
		DllExport virtual ~CTrainEdgePrior(void);

		DllExport virtual void	reset(void);	

		DllExport virtual void	addFeatureVecs(const Mat &featureVector1, byte gt1, const Mat &featureVector2, byte gt2);
		DllExport virtual void	train(bool doClean = false);

	protected:
		DllExport virtual void 	saveFile(FILE *pFile) const;
		DllExport virtual void 	loadFile(FILE *pFile);		
		/**
		* @brief Calculates the edge potential, based on the feature vectors

		* @details This function returns matrix with diagonal elements equal to parameter \f$\vec{\theta}\f$ provided through argument \b params 
		* penalized with a penalization function \f$\mathcal{P}\f$, which depends on the difference between arguments \b featureVector1 and \b featureVector2
		* and multiplied with the corresponding edge prior probability; 
		* all the other elements are the edge prior probability matrix values: 
		* \f[edgePot[nStates][nStates] = \begin{bmatrix} \theta_1\cdot\mathcal{P}\cdot p_{1,1} & p_{1,2} & \cdots & p_{1,n} \\ 
		* p_{2,1} & \theta_2\cdot\mathcal{P}\cdot p_{2,2} & \cdots & p_{2,n} \\ 
		* \vdots & \vdots & \ddots & \vdots \\ 
		* p_{n,1} & p_{n,2} & \cdots & \theta_n\cdot\mathcal{P}\cdot p_{n,n} \end{bmatrix}, \f]
		* where \f$p_{i,j}\f$ is the normalized according to @ref ePotNormApproach prior probability of edge to connect nodes with states (classes) \a i and \a j,
		* \f$\mathcal{P}\equiv\mathcal{P}(d;\,\lambda)\f$ is the penalization function, where \f$d = ||\textbf{f}_1 - \textbf{f}_2||_2\f$ is the Euclidean distance between feature vectors,  
		* and \f$\lambda\f$ is a parameter provided through argument \b params. For more details on penalization fuction, please refere to @ref ePotPenalApproach.
		* @param featureVector1 Multi-dimensinal point \f$\textbf{f}_1\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the first node of the edge
		* @param featureVector2 Multi-dimensinal point \f$\textbf{f}_2\f$: Mat(size: nFeatures x 1; type: CV_{XX}C1), corresponding to the second node of the edge
		* @param vParams Array of control parameters \f$\{\vec{\theta},\lambda\} \f$. \f$\vec{\theta}\f$ may consist either from \a one parameter (in this case all the diagonal elemets will be the same), 
		* or from \a nStates parameters, specifying smoothness strength for each state (class) individually; \f$\lambda\f$ consists from \a one parameter.
		* @return The edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual Mat	calculateEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, const vec_float_t &vParams) const;


	private:
		inline void				loadPriorMatrix(void);
		Mat						m_prior;					// = Mat();	// The prior matrix
	};
}