//
// Created by ahambasan on 22.02.20.
//

#pragma once

#include "ParamEstimation.h"

namespace DirectGraphicalModels {
	// ================================ Particle Swarm Optimization Class ===============================
	/**
	* @ingroup moduleParamEst
	* @brief The Particle Swarm Optimization search method class
	* @details The Particle Swarm Optimization search method is an iterative optimisation algorithm that does not require an estimate for the gradient of the objective function:
	* \f$ f:\mathbb{R}^n\rightarrow\mathbb{R} \f$, where \f$ n \f$ is the number of parameters (arguments). In order to find the extremum point, one may use the
	* common case:
	* @anchor pso_example_code
	* @code
	* using namespace DirectGraphicalModels;
	*
	* const word nParams = 2;
	*
	* const vec_float_t	vInitParams = {0.0f, 0.0f};		// coordinates of the initial point for the search algorithm

	* CParamEstimationPSO CParamEstimationPSO(vInitParams);
	* vec_float_t vParams = pso.getParams(objectiveFunction);

	* @endcode
	* @author Alexandru Hambasan, a.hambasan@jacobs-university.de
	*/

	class CParamEstimationPSO : public CParamEstimation
	{
	private:
		/**
		 * @brief Structure for representing a bird like object (BOID)
		 */
		struct Boid {
			vec_float_t vArgBest;               // personal best parameters			// Initialized to the given by the user parameters
			float		valBest;				// the value of the objective function for the personal best parameters
			vec_float_t vArgCurrent;            // personal position parameters		// Initialized to be random in (-10; 10)
			float		valCurrent;				// the value of the objective function for the personal position parameters
			vec_float_t vVelocity;              // velocity of the particle/ BOID	// Initialized with 1
		};

		const size_t NUMBER_BOIDS       = 500;  // number of particles/ boids
		const size_t MAX_NR_ITERATIONS  = 1000; // number of iterations

		const float C1_DEFAULT_VALUE    = 1.7f; // default value for cognitive component parameter
		const float C2_DEFAULT_VALUE    = 1.5f; // default value for social component parameter
		const float W_DEFAULT_VALUE     = 0.5f; // default value for inertia parameter


		std::vector<Boid>	m_vBoids;           // vector containing the particles/ boids
		float				m_c1;               // cognitive component parameter
		float				m_c2;               // social component parameter
		float				m_w;                // inertia parameter

		vec_float_t 		m_vGlobalArgBest;   // global best parameters
		float				m_globalValBest = -101;	// Value of the objective function for global best parameters

		// TODO: remove it
		size_t m_iteration = 0;
		bool m_ifFirstCall = true;

	public:
		/**
		 * @brief Constructor
		 * @param nParams Number of parameters (arguments) of the objective function
		 */
		DllExport CParamEstimationPSO(size_t nParams);
		DllExport ~CParamEstimationPSO(void) = default;

		DllExport virtual void			reset(void) override;
		DllExport virtual vec_float_t	getParams(float val) override;                     
		DllExport virtual bool			isConverged(void) const override; 
		
		/**
		 * @brief Sets gBest variable to the best parameters founds
		 * @details This function updates the global best parameters (arguments) of the objective function
		 * based on its outcome value \b objectiveFunct_val
		 * (See [example code](#pso_example_code) for more details)
		 * @param objectiveFunct The objective function to be minimized.
		 */
		DllExport vec_float_t   getParams(const std::function<float(vec_float_t)>& objectiveFunct);
	};
}
