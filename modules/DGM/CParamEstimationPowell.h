// The CParamEstimationPowell search method class for random model parameters trainig
// Written by Sergey G. Kosov in 2013, 2016 for Project X
#pragma once

#include "CParamEstAlgorithm.h"

namespace DirectGraphicalModels
{
	// ================================ CParamEstimationPowell Class ===============================
	/**
	* @ingroup moduleParamEst
	* @brief The CParamEstimationPowell search method class
	* @details The CParamEstimationPowell search method is an iterative optimisation algortihm that does not require an estimate for the gradient of the objective function:
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

	* CParamEstimationPowell powell(nParams);
	* powell.setInitParams(vInitParams);
	* powell.setDeltas(vInitDeltas);

	* while(!powell.isConverged()) {
	* 	float val = objectiveFunction(vParams);
	* 	vParams = powell.getParams(val);
	* } 
	* @endcode
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CParamEstimationPowell : public CParamEstAlgorithm
	{
	public:
		/**
		 * @brief Constructor
		 * @param nParams Number of parameters (arguments) of the objective function
		 */		
		DllExport CParamEstimationPowell(size_t nParams);
		DllExport virtual ~CParamEstimationPowell(void) = default;

		DllExport virtual void			reset(void) override;
		/**
		 * @brief Gets the updated parameters (arguments)
		 * @details This function updates the parameters (arguments) of the objective function based on its outcome value \b val and retunrs them
		 * (See [example code](#powell_example_code) for more details)
		 * @param val The current value of the objective function
		 * @return The pointer to array with the updated parameters
		 */
		DllExport virtual vec_float_t	getParams(float val) override;
		DllExport virtual vec_float_t	getParams(std::function<float(vec_float_t)> objectiveFunct) override;
		DllExport virtual bool			isConverged(void) const override;
		
		/**
		 * @brief Sets the acceleration coefficient
		 * @details Incrasing this parameter may speed-up the convergence of the method, however too large values may affect the calculation stability
		 * > Default value is \b 0.1
		 * @param acceleration The acceleration coefficient
		 */
		DllExport void					setAcceleration(float acceleration);


	private:
		size_t		m_paramID;		// index of a currently adjusting argument
		size_t		m_nSteps;		// number of adjustments for one argument
		float		m_midPoint;		// parameter value for kappa: 0
		float		m_koeff;		// koefficient for optimized CParamEstimationPowell search method
		float		m_acceleration;	// acceleration of search along one direction
		
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


