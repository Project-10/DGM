//
// Created by ahambasan on 22.02.20.
//
#include "ParamEstimationPSO.h"
#include "random.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
    CParamEstimationPSO::CParamEstimationPSO(size_t nParams)
        : CParamEstimation(nParams)
		, m_vBoids(NUMBER_BOIDS)
    {
        m_gBest = m_vParams;			// m_vParams is empty. May be an error here. This is definitely should be in the setInitParams() function
		
        reset();
    }

    void CParamEstimationPSO::reset()
	{
        // initialize meta parameters
        m_c1 = C1_DEFAULT_VALUE;
        m_c2 = C2_DEFAULT_VALUE;
        m_w  = W_DEFAULT_VALUE;

        // initialize boids
        for (Boid& boid : m_vBoids) {
			boid.vParams.resize(m_vParams.size());
			for (float& p : boid.vParams)
				p = random::U(-10.0f, 10.0f);
			
			boid.vVelocity.resize(m_vParams.size());
			std::fill(boid.vVelocity.begin(), boid.vVelocity.end(), 1.0f);
            
			boid.vBest = m_vParams;		// m_vParams is empty. May be an error here.
        }
		
        std::fill(m_vMin.begin(), m_vMin.end(), -FLT_MAX);
        std::fill(m_vMax.begin(), m_vMax.end(), FLT_MAX);
        std::fill(m_vParams.begin(), m_vParams.end(), 0.0f);
    }

	// TODO: implement this function
	vec_float_t CParamEstimationPSO::getParams(float val) {
		return m_gBest;
	}

	// TODO: implement this function
	bool CParamEstimationPSO::isConverged(void) const {
		return true;
	}


    vec_float_t CParamEstimationPSO::getParams(const std::function<float(vec_float_t)>& objectiveFunct)
	{
		// Why it is called MAX_NR_ITERATIONS? There is no break from the loop, so no chance to
		// stop iterations earlier. NUMBER_ITERATIONS is more appropriate name for this. Next, if the method converges at iteration
		// it = NUMBER_ITERATIONS / 10. Does it mean that we would waste 9 * NUMBER_ITERATIONS * 3 * NUMBER_BOIDS / 10 calls of the objective
		// function?  I think this class should use definetely isConverged() function.
		for (size_t it = 0; it < MAX_NR_ITERATIONS; it++) {	// <= these iterations will be performed outside the function
            // In this loop the objective function is called (3 * NUMBER_BOIDS) times. Is it possible to reduce this amount?
			// 1 call with boid.vParams
			// 2 call with boid.vBest
			// 3 call with m_gBest
			// the boid.vBest and m_gBest are updated inside the loop, that means that the objective function was already called for them.
			for (auto& boid : m_vBoids) {
                float objectiveFunct_val = objectiveFunct(boid.vParams);	// <= objectiveFunct_val will be achieved from an argument of the function
                float pBest_val = objectiveFunct(boid.vBest);				// <= pBest_val will be achieved from an argument of the function
                if (objectiveFunct_val < pBest_val)
                    boid.vBest = boid.vParams;

                float gBest_val = objectiveFunct(m_gBest);					// <= gBest_val will be achieved from an argument of the function
                if (objectiveFunct_val < gBest_val)
                    m_gBest = boid.vBest;
            }
            
			// Update vVelocity and vParams of every boid
			for (Boid& boid : m_vBoids) {
                float r1 = random::U<float>();
                float r2 = random::U<float>();
                for (auto d = 0; d < m_vParams.size(); d++) {
                    boid.vVelocity[d] = m_w * boid.vVelocity[d] + m_c1 * r1 * (boid.vBest[d] - boid.vParams[d]) +
                                        m_c2 * r2 * (m_gBest[d] - boid.vParams[d]);
                    boid.vParams[d] = boid.vParams[d] + boid.vVelocity[d];	// I think velocity is the same as deltas in Powell. Maybe re-use the corresponding container
                }
            }
        }
		return m_gBest;
    }

}
