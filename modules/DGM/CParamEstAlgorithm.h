//
// Created by ahambasan on 21.02.20.
//
#pragma once

#include "types.h"

namespace DirectGraphicalModels 
{
	/**
	 *
	 */
	class CParamEstAlgorithm
	{
	public:
		/**
		 * @brief Constructor
		 * @param nParams Number of parameters (arguments) of the objective function
		 */
		DllExport CParamEstAlgorithm(size_t nParams);
		DllExport CParamEstAlgorithm(const CParamEstAlgorithm&) = delete;
		DllExport virtual ~CParamEstAlgorithm() = default;
		DllExport const CParamEstAlgorithm& operator=(const CParamEstAlgorithm&) = delete;

		/**
		 * @brief Resets class variables
		 */
		DllExport virtual void reset() = 0;
		/**
		 * @brief Gets the updated parameters (arguments)
		 * @details This function updates the parameters (arguments) of the objective function based on its outcome value \b val and retunrs them
		 * (See [example code](#powell_example_code) for more details)
		 * @param val The current value of the objective function
		 * @return The array with the updated parameters
		 */
		DllExport virtual vec_float_t getParams(float val) = 0;
		/**
		 * @param objectiveFunct The objective function to be minimized
		 * @return Array of the best parameters found
		 */
		DllExport virtual vec_float_t getParams(std::function<float(vec_float_t)> objectiveFunction) = 0;
		/**
		 * @brief Indicates weather the method has converged
		 * @retval true if the method has converged
		 * @retval false otherwise
		 */
		DllExport virtual bool isConverged(void) const = 0;
		
		/**
		 * @brief Sets the initial parameters (arguments) for the search algorithm
		 * @details
		 * > Default values are \b 0 for all parameters (arguments)
		 * @param vParams An array with the initial values for the search algorithm
		 */
		DllExport void	setInitParams(const vec_float_t& vParams);
		/**
		* @brief Sets the searching steps along the parameters (arguments)
		* @details
		* > Default values are \b 0.1 for all parameters (arguments)
		* @param vDeltas An array with the offset values for each parameter (argument)
		*/
		DllExport void	setDeltas(const vec_float_t& vDeltas);
		/**
		 * @brief Sets the lower boundary for parameters (arguments) search
		 * @details
		 * > Default values are \f$-\infty\f$ for all parameters (arguments)
		 * @param vMinParam An array with the minimal parameter (argument) values
		 */
		DllExport void	setMinParams(const vec_float_t& vMinParam);
		/**
		 * @brief Sets the upper boundary for parameters (arguments) search
		 * @details
		 * > Default values are \f$+\infty\f$ for all parameters (arguments)
		 * @param vMaxParam An array with the maximal parameter (argument) values
		 */
		DllExport void	setMaxParams(const vec_float_t& vMaxParam);


		protected:
			vec_float_t m_vParams;                  ///< array of the initial parameter parameters
			vec_float_t	m_vDeltas;					///< array of the delta values for each parameter
			vec_float_t m_vMin;                     ///< array of minimal parameter values
			vec_float_t m_vMax;                     ///< array of maximal parameter values
	};
}