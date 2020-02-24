//
// Created by ahambasan on 21.02.20.
//

#ifndef DGM_PARAMESTALGORITHM_H
#define DGM_PARAMESTALGORITHM_H

#include "types.h"

namespace DirectGraphicalModels {
    class ParamEstAlgorithm {
    public:
        /**
         * @brief Resets class variables
         */
        DllExport virtual void reset() = 0;
        /**
        * @brief Sets the initial parameters (arguments) for the search algorithm
        * @details
        * > Default values are \b 0 for all parameters (arguments)
        * @param vParams An array with the initial values for the search algorithm
        */
        DllExport virtual void setInitParams(const vec_float_t &vParams) = 0;
        /**
        * @brief Sets the searching steps along the parameters (arguments)
        * @details
        * > Default values are \b 0.1 for all parameters (arguments)
        * @param vDeltas An array with the offset values for each parameter (argument)
        */
        DllExport virtual void setDeltas(const vec_float_t &vDeltas) {}
        /**
        * @brief Sets the lower boundary for parameters (arguments) search
        * @details
        * > Default values are \f$-\infty\f$ for all parameters (arguments)
        * @param vMinParam An array with the minimal parameter (argument) values
        */
        DllExport virtual void setMinParams(const vec_float_t &vMinParam) = 0;
        /**
        * @brief Sets the upper boundary for parameters (arguments) search
        * @details
        * > Default values are \f$+\infty\f$ for all parameters (arguments)
        * @param vMaxParam An array with the maximal parameter (argument) values
        */
        DllExport virtual void setMaxParams(const vec_float_t &vMaxParam) = 0;
        /**
        * @brief Sets the acceleration coefficient
        * @details Incrasing this parameter may speed-up the convergence of the method, however too large values may affect the calculation stability
        * > Default value is \b 0.1
        * @param acceleration The acceleration coefficient
        */
        DllExport virtual void setAcceleration(float acceleration) {}
        /**
        * @brief Gets the updated parameters (arguments)
        * @details This function updates the parameters (arguments) of the objective function based on its outcome value \b val and retunrs them
        * (See [example code](#powell_example_code) for more details)
        * @param val The current value of the objective function
        * @return The pointer to array with the updated parameters
        */
        DllExport virtual vec_float_t getParams(float val) {}

        DllExport virtual vec_float_t getParams(float (*objectiveFunct)(vec_float_t)) = 0;
        /**
        * @brief Indicates weather the method has converged
        * @retval true if the method has converged
        * @retval false otherwise
        */
        DllExport virtual bool isConverged() {}
    };
}



#endif //DGM_PARAMESTALGORITHM_H
