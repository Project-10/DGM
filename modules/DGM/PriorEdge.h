// Edge prior probability estimation class
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "Prior.h"

namespace DirectGraphicalModels
{
	/**
	* @brief Normalization approach flag
	* @details These flags specify the  approach for normalization of the edge potential matrix. The difference between symmetric and asymmetric approaches:
	* - \b Standard \b approach performs a per-element multiplication of the histogram matrix with the calar value \f$1/sum\f$, where \f$sum\f$ - is the sum of all matrix elements.
	* - Let \f$A,B\in\mathbb{S}\f$ - are any two different states (classes) and \f$N_{AA}\f$ (and, consequently \f$N_{BB}\f$) - are number of within class transitions in 
	* training data, \a i.e. the doubled number of neighboring sites with the same state (class). And \f$N_{AB}\equiv N_{BA}\f$ - are the number of 
	* transitions between classes \f$A\f$ and \f$B\f$, \a i.e. the number of neighboring sites with states (classes) \f$A\f$ and \f$B\f$.
	* - \b Symmetric \b approach after the normalization guaranties that resulting \b ePot matrix is symmetric, \a i.e. \f$M_{AB} = M_{BA}\f$ and \f$M_{AA} = M_{BB} = 1\f$. 
	* In case of \f$N_{AA}, N_{BB} > N_{AB}\equiv N_{BA}\f$, the largest number in \b ePot matrix will be \f$1\equiv M_{AA}\equiv M_{BB}\f$ (ones on the diagonal); 
	* and in case of \f$N_{AB}\cdot N_{AB}\geq N_{AA}\cdot N_{BB}\f$, it is possible to have values larger than \f$M_{AB} > 1\f$. This case is possible to model for 
	* academic purpose, but it is almost improbable in real life applications.
	* - \b Asymmetric \b approach takes into account the difference in number of occurrences of states (classes) in the training data (\a i.e. prior 
	* probabilities \f$p(A)\f$ and \f$p(B)\f$). After the normalization it guaranties that the largest value in \b ePot matrix does not exceed \f$1\f$. The 
	* case, when the diagonal values of the matrix are not the largest is usually probable (\a e.g. \f$1 = AB > AA\f$ after the normalization). 
	* This may lead to erosion of areas with small occurrence rate.
	* - The result of asymmetric approach could be approximated from the result of symmetric approach, by multiplying the columns 
	* of the \b ePot matrix with the corresponding to the base columns state (class) the prior state (class) probability.
	*/
	enum ePotNormApproach {
		eP_APP_NORM_STANDARD,	///< Standard approach
		eP_APP_NORM_SYMMETRIC,	///< Symmetric approach
		eP_APP_NORM_ASYMMETRIC	///< Asymmetric approach
	}; 	

// ================================ Edge Prior Class ================================
/**
@brief %Edge prior probability estimation class.
@author Sergey G. Kosov, sergey.kosov@project-10.de
*/	
	class CPriorEdge : public CPrior
	{
	public:
/**
@brief Constructor
@param nStates Number of states (classes)
@param normApp Flag specifying the co-occurance histogram matrix normalization approach (Ref. @ref ePotNormApproach)
*/
		DllExport CPriorEdge(byte nStates, ePotNormApproach normApp = eP_APP_NORM_SYMMETRIC) : CPrior(nStates, RM_PAIRWISE), CBaseRandomModel(nStates), m_normApp(normApp) {}
		DllExport ~CPriorEdge(void) {}

/**
@brief Adds the groud-truth value to the co-occurance histogram matrix
@details Here \b gt1 is the X-coordinate of the co-occurance histogram matrix and \b gt2 - Y-coordinate of the co-occurance histogram matrix.
@param gt1 The ground-truth state (value) of the first node in edge. 
@param gt2 The ground-truth state (class) of the second node in edge. 
*/
		DllExport void			addEdgeGroundTruth(byte gt1, byte gt2); 

		
		
	protected:		
/**
@brief Returns the prior edge probability
@details This function returns the normalized class co-occurance histogram, which ought to be build during the training phase with help of the addEdgeGroundTruth() function.
If the histogram was not built, this functions returns a uniform distribution "all ones".
@return Prior edge probability matrix: Mat(size: nStates x nStates; type: CV_32FC1)
*/
		DllExport Mat			calculatePrior(void) const;


	protected:
		ePotNormApproach		m_normApp;		///< Flag specifying the co-occurance histogram matrix normalization approach (Ref. @ref ePotNormApproach)

	};
}