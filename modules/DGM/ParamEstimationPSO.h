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
			vec_float_t vArgBest;                   // personal best parameters			// Initialized to the given by the user parameters
			float		valBest;				    // the value of the objective function for the personal best parameters
			vec_float_t vArgCurrent;                // personal position parameters		// Initialized to be random in (-10; 10)
			std::pair<float, bool>	valCurrent;		// pair of the value of the objective function for the personal position parameters and its status
			vec_float_t vVelocity;                  // velocity of the particle/ BOID	// Initialized with 1
			bool hasConverged = false;
		};

		const size_t NUMBER_BOIDS       = 500;      // number of particles/ boids

		const float C1_DEFAULT_VALUE    = 1.49617f; // default value for cognitive component parameter
		const float C2_DEFAULT_VALUE    = 1.49617f; // default value for social component parameter
		const float W_DEFAULT_VALUE     = 0.72984f; // default value for inertia parameter

		std::vector<Boid>	m_vBoids;               // vector containing the particles/ boids
		float				m_c1;                   // cognitive component parameter
		float				m_c2;                   // social component parameter
		float				m_w;                    // inertia parameter

		float				m_globalValBest;	    // Value of the objective function for global best parameters

		const float         UNINITIALIZED = -INFINITY;

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

	};
}
