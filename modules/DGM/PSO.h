//
// Created by ahambasan on 22.02.20.
//

#pragma once

#include "ParamEstAlgorithm.h"

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

	* PSO PSO(vInitParams);
	* vec_float_t vParams = pso.getParams(objectiveFunction);

	* @endcode
	* @author Alexandru Hambasan, a.hambasan@jacobs-university.de
	*/

	class PSO : public CParamEstAlgorithm 
	{
	private:
		/**
		 * @brief Structure for representing a bird like object (BOID)
		 */
		struct Boid {
			vec_float_t pBest;                  // personal best parameters
			vec_float_t velocity;               // velocity of the particle/ BOID
			vec_float_t pParams;                // personal position parameters
		};

		const size_t NUMBER_BOIDS       = 500;  // number of particles/ boids
		const size_t MAX_NR_ITERATIONS  = 1000; // number of iterations

		const float C1_DEFAULT_VALUE    = 1.7f; // default value for cognitive component parameter
		const float C2_DEFAULT_VALUE    = 1.5f; // default value for social component parameter
		const float W_DEFAULT_VALUE     = 0.5f; // default value for inertia parameter


		std::vector<Boid> m_vBoids;                  // vector containing the particles/ boids
		float             c1;                   // cognitive component parameter
		float             c2;                   // social component parameter
		float             w;                    // inertia parameter

		vec_float_t gBest;                      // global best parameters

		bool        isThreadsEnabled;           // boolean afferent to multiThreading enabling

		std::mutex  mtx;                        // mutex for multiThreading option

		/**
	  * @brief Sets gBest variable to the best parameters founds
	  * @details This function updates the global best parameters (arguments) of the objective function
	  * based on its outcome value \b objectiveFunct_val
	  * (See [example code](#pso_example_code) for more details)
	  * @param objectiveFunct The objective function to be minimized.
	  */
		void runPSO(const std::function<float(vec_float_t)>& objectiveFunct);
		void runPSO_withThreads(const std::function<float(vec_float_t)>& objectiveFunct, size_t idx);

	public:
		/**
		 * @brief Default Constructor - initialize the parameter vector of the objective function to
		 */
		DllExport PSO(size_t nParams);
		/**
		 * @brief Argument Constructor
		 * @param vParams An array containing the initial value of the parameters to be optimized
		 */
		DllExport explicit PSO(const vec_float_t &vParams);
		/**
		 * @brief Destructor
		 */
		DllExport ~PSO() = default;

		DllExport virtual void			reset(void) override;
		DllExport virtual vec_float_t	getParams(float val) override { return m_vParams; }                     // TODO: implement this function
		DllExport virtual vec_float_t   getParams(std::function<float(vec_float_t)> objectiveFunct) override;
		DllExport virtual bool			isConverged(void) const override { return true; }                       // TODO: implement this function
		
		/**
		 * @return boolean value representing whether multiThreading is enabled or not
		 */
		DllExport bool isMultiThreadingEnabled() const;

		/**
		 * @brief Enables multiThreading
		 */
		DllExport void enableMultiThreading();

		/**
		 * @brief Overloaded function for convenience
		 * @param enable Boolean parameter for enabling/ disabling multiThreading
		 */
		DllExport void enableMultiThreading(bool enable);

		/**
		 * @brief Disables multiThreading
		 */
		DllExport void disableMultiThreading();
	};
}