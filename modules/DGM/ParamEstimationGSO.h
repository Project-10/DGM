//
//  created by sabyrrakhim06 on 10.05.2021
//  inspired by PSO algorithm implemented by ahambasan and sabyrrakhim06
//

#pragma once

#include "ParamEstimation.h"

namespace DirectGraphicalModels {

class CParamEstimationGSO : public CParamEstimation {
private:
    
    struct Boid {
    public:
        vec_float_t vArgBest;                   // personal best parameters            // Initialized to the given by the user parameters
        float        valBest;                    // the value of the objective function for the personal best parameters
        vec_float_t vArgCurrent;                // personal position parameters        // Initialized to be random in (-10; 10)
        std::pair<float, bool>    valCurrent;        // pair of the value of the objective function for the personal position parameters and its status
        vec_float_t vVelocity;                  // velocity of the particle/ BOID    // Initialized with 1
        vec_float_t vGlobalVal;
        bool hasConverged = false;
        //std::vector<std::shared_ptr<Boid>> m_vpBoids; // shared pointer of each sub-swarm
    };
    
    struct GBoid {
        vec_float_t vGalBest;
        float        galBest;
        vec_float_t vGalCurrent;
        std::pair<float, bool>   galCurrent;
        vec_float_t vGVelocity;
        bool hasConverged_G = false;
    };
    
    const size_t NUMBER_BOIDS = 100;    // number of boids
    const size_t NUMBER_SUBSWARMS = 20;    // number of subswarms
    const size_t SIZE_SUBSWARM = 5;     // size of each subswarm
    const size_t L1 = 280;
    const size_t L2 = 1500;
    const size_t EPOCH_MAX = 5;
    
    // initialize hyperparameters
    const float C1_DEFAULT_VALUE = 2.05f;   // first cognitive parameter
    const float C2_DEFAULT_VALUE = 2.05f;   // first social hyperparameter
    const float C3_DEFAULT_VALUE = 2.05f;   // second cognitive hyperparameter
    const float C4_DEFAULT_VALUE = 2.05f;   // second social hyperparameter
    const float W_DEFAULT_VALUE  =  0.72984f; // default value of inertia hyperparameter
    
    std::vector<Boid>     m_vBoids;     // vector containing the particles/ boids
    std::vector<GBoid>    m_vGboids;  // vector containing subswarms
    //  cv::Mat m_mSuperSwarm = cv::Mat(NUMBER_SUBSWARMS, m_vParams.size(), int CV_64FC1, float()); // vector containing subswarm optima
    float                 m_c1;
    float                 m_c2;
    float                 m_c3;
    float                 m_c4;
    float                 m_w;
    float                 m_w1;
    float                 m_w2;
    
    float m_globalValBest;  // global best value
    float m_globalGalBest;  // galactic best value
    
    const float UNINITIALIZED = -INFINITY;
    
public:
    DllExport CParamEstimationGSO(size_t nParams);
    DllExport ~CParamEstimationGSO(void) = default;
    
    DllExport virtual void            reset(void) override;
    DllExport virtual vec_float_t     getParams(float val) override;
    DllExport virtual bool            isConverged(void) const override;
};
}

