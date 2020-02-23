// The Powell search method class for random model parameters trainig
// Written by Sergey G. Kosov in 2013, 2016 for Project X
#pragma once

#include "ParamEstAlgorithm.h"
#include "types.h"

namespace DirectGraphicalModels
{
	// ================================ Powell Class ===============================
	/**
	* @ingroup moduleParamEst
	* @brief The Powell search method class
	* @details The Powell search method is an iterative optimisation algortihm that does not require an estimate for the gradient of the objective function: 
	* \f$ f:\mathbb{R}^n\rightarrow\mathbb{R} \f$, where \f$ n \f$ is the number of parameters (arguments). In order to find the extremum point, one may use the
	* common case:
	* @anchor powell_example_code
	* @code
	* using namespace DirectGraphicalModels;
	* 
	* const word nParams = 2;
	*
	* const vec_float_t	vInitParams = {0.0f, 0.0f};		// coordinates of the initial point for the search algorithm
	* const vec_float_t	vInitDeltas = {0.1f, 0.1f};		// searching steps along the parameters (arguments)
	* vec_float_t vParams = vInitParams;

	* CPowell powell(nParams);
	* powell.setInitParams(vInitParams);
	* powell.setDeltas(vInitDeltas);

	* while(!powell.isConverged()) {
	* 	float val = objectiveFunction(vParams);
	* 	vParams = powell.getParams(val);
	* } 
	* @endcode
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CPowell : public ParamEstAlgorithm
	{
	public:
		/**
		* @brief Constructor
		* @param nParams Number of parameters (arguments) of the objective function
		*/		
		DllExport CPowell(size_t nParams);
		DllExport CPowell(const CPowell&) = delete;
		DllExport ~CPowell(void) = default;
		DllExport const CPowell& operator=(const CPowell&) = delete;

		/**
		* @brief Resets class variables
		*/
		DllExport void	  reset(void) override;
		/**
		* @brief Sets the initial parameters (arguments) for the search algorithm
		* @details 
		* > Default values are \b 0 for all parameters (arguments)
		* @param vParams An array with the initial values for the search algorithm
		*/
		DllExport void	  setInitParams(const vec_float_t& vParams) override;
		/**
		* @brief Sets the searching steps along the parameters (arguments)
		* @details 
		* > Default values are \b 0.1 for all parameters (arguments)
		* @param vDeltas An array with the offset values for each parameter (argument)
		*/
		DllExport void	  setDeltas(const vec_float_t& vDeltas) override;
		/**
		* @brief Sets the lower boundary for parameters (arguments) search
		* @details
		* > Default values are \f$-\infty\f$ for all parameters (arguments)
		* @param vMinParam An array with the minimal parameter (argument) values
		*/
		DllExport void	  setMinParams(const vec_float_t& vMinParam) override;
		/**
		* @brief Sets the upper boundary for parameters (arguments) search
		* @details
		* > Default values are \f$+\infty\f$ for all parameters (arguments)
		* @param vMaxParam An array with the maximal parameter (argument) values
		*/
		DllExport void	  setMaxParams(const vec_float_t& vMaxParam) override;
		/**
		* @brief Sets the acceleration coefficient
		* @details Incrasing this parameter may speed-up the convergence of the method, however too large values may affect the calculation stability
		* > Default value is \b 0.1
		* @param acceleration The acceleration coefficient
		*/
		DllExport void	  setAcceleration(float acceleration) override;
		/**
		* @brief Gets the updated parameters (arguments)
		* @details This function updates the parameters (arguments) of the objective function based on its outcome value \b val and retunrs them 
		* (See [example code](#powell_example_code) for more details)
		* @param val The current value of the objective function
		* @return The pointer to array with the updated parameters
		*/
		DllExport vec_float_t getParams(float val) override;
		/**
		* @brief Indicates weather the method has converged
		* @retval true if the method has converged
		* @retval false otherwise
		*/
		DllExport bool	  isConverged(void) const;

        /**
        * @param objectiveFunct The objective function to be minimized
        * @return Array of the best parameters founds
        */
        DllExport vec_float_t getParams(float (*objectiveFunct)(vec_float_t)) override;


	private:
		size_t		m_nParams;		// number of parameters (arguments of the objective function)
		size_t		m_paramID;		// index of a currently adjusting argument
		size_t		m_nSteps;		// number of adjustments for one argument
		float		m_midPoint;		// parameter value for kappa: 0
		float		m_koeff;		// koefficient for optimized Powell search method
		float		m_acceleration;	// acceleration of search along one direction
		
		vec_float_t m_vParams;		// array of the parameters
		vec_float_t	m_vDeltas;		// array of the delta values for each parameter
		vec_float_t	m_vMin;			// array of minimal parameter values
		vec_float_t	m_vMax;			// array of maximal parameter values
		vec_float_t	m_vKappa;		// method's auxilary array 
		vec_bool_t	m_vConverged;	// array of flags, indicating converged variables


		// Simplified accessors for current argument
		#define curArg m_vParams[m_paramID]
		#define delta m_vDeltas[m_paramID] 
		#define minArg m_vMin[m_paramID]
		#define maxArg m_vMax[m_paramID]
		#define convArg m_vConverged[m_paramID]


	private:		
		/// coordinates of the Kappa function
		enum {
			mD,		///< minus Delta
			oD,		///< null Delta (current position)
			pD		///< plus Delta
		};
	};

}


