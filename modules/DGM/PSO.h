//
// Created by ahambasan on 22.02.20.
//

#ifndef DGM_PSO_H
#define DGM_PSO_H

#include "ParamEstAlgorithm.h"
#include <vector>

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

    class PSO : public ParamEstAlgorithm {
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

        const float C1_DEFAULT_VALUE    = 1.7;  // default value for cognitive component parameter
        const float C2_DEFAULT_VALUE    = 1.5;  // default value for social component parameter
        const float W_DEFAULT_VALUE     = 0.5;  // default value for inertia parameter


        std::vector<Boid> b_n;                  // vector containing the particles/ boids
        float             c1{};                   // cognitive component parameter
        float             c2{};                   // social component parameter
        float             w{};                    // inertia parameter


        vec_float_t m_vParams;                  // array of the initial parameter parameters
        vec_float_t gBest;                      // global best parameters

        vec_float_t m_vMin;                     // array of minimal parameter values
        vec_float_t m_vMax;                     // array of maximal parameter values
        vec_bool_t  m_vConverged;
        size_t      m_nParams{};                  // number of parameters (arguments of the objective function)

        bool        isThreadsEnabled{};           // boolean afferent to multiThreading enabling

        std::mutex  mtx;                        // mutex for multiThreading option

        /**
      * @brief Sets gBest variable to the best parameters founds
      * @details This function updates the global best parameters (arguments) of the objective function
      * based on its outcome value \b objectiveFunct_val
      * (See [example code](#pso_example_code) for more details)
      * @param objectiveFunct The objective function to be minimized.
      */
        void runPSO(float (*objectiveFunct)(vec_float_t));
        void runPSO_withThreads(float (*objectiveFunct)(vec_float_t), size_t idx);

    public:
        /**
         * @brief Default Constructor - initialize the parameter vector of the objective function to
         */
        DllExport PSO();
        /**
       * @brief Argument Constructor
       * @param vParams An array containing the initial value of the parameters to be optimized
       */
        DllExport explicit PSO(const vec_float_t &vParams);

        /**
         * @brief Resets class variables
         */
        DllExport void reset() override;

        /**
        * @brief Sets the initial parameters (arguments) for the search algorithm
        * @details
        * > Default values are \b 0 for all parameters (arguments)
        * @param vParams An array with the initial values for the search algorithm
        */
        DllExport void setInitParams(const vec_float_t &vParams) override;

        /**
        * @brief Sets the lower boundary for parameters (arguments) search
        * @details
        * > Default values are \f$-\infty\f$ for all parameters (arguments)
        * @param vMinParam An array with the minimal parameter (argument) values
        */
        DllExport void setMinParams(const vec_float_t &vMinParam) override;

        /**
       * @brief Sets the upper boundary for parameters (arguments) search
       * @details
       * > Default values are \f$+\infty\f$ for all parameters (arguments)
       * @param vMaxParam An array with the maximal parameter (argument) values
       */
        DllExport void setMaxParams(const vec_float_t &vMaxParam) override;

        /**
         * @param objectiveFunct The objective function to be minimized
         * @return Array of the best parameters founds
         */
        DllExport vec_float_t getParams(float (*objectiveFunct)(vec_float_t)) override;

        /**
         * @return boolean value representing whether multiThreading is enabled or not
         */
        DllExport bool isMultiThreadingEnabled();

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

        /**
        * @brief Indicates weather the method has converged
        * @retval true if the method has converged
        * @retval false otherwise
        */
        DllExport bool isConverged() const;

        /**
         * @brief Destructor
         */
        DllExport ~PSO();

    };
}


#endif //DGM_PSO_H
