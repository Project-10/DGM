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
        m_vGlobalArgBest = m_vParams;			// m_vParams is empty. May be an error here. This is definitely should be in the setInitParams() function
		
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
            boid.vArgCurrent.resize(m_vParams.size());
			for (float& p : boid.vArgCurrent)
				p = random::U(-10.0f, 10.0f);
            boid.valCurrent = -101;
			
            boid.vArgBest = m_vParams;		// m_vParams is empty. May be an error here.
            boid.valBest = -101;
            
            boid.vVelocity.resize(m_vParams.size());
			std::fill(boid.vVelocity.begin(), boid.vVelocity.end(), 1.0f);
        }
		
        std::fill(m_vMin.begin(), m_vMin.end(), -FLT_MAX);
        std::fill(m_vMax.begin(), m_vMax.end(), FLT_MAX);
        std::fill(m_vParams.begin(), m_vParams.end(), 0.0f);
    }

	// TODO: implement this function
	vec_float_t CParamEstimationPSO::getParams(float val) 
    {
        // On the first call we get the value for the m_globalValBest and boid.vArgBest 
        if (m_ifFirstCall) {
            m_globalValBest = val;
            for (Boid& boid : m_vBoids)
                boid.valBest = val;
            m_ifFirstCall = false;
        }

        for (Boid& boid : m_vBoids) {
            
            // Boid: Value Current
            if (boid.valCurrent < -100) {
                boid.valCurrent = -1;
                return boid.vArgCurrent;
            }
            else if (boid.valCurrent < 0)
                boid.valCurrent = val;

            // Boid: Value Best     // TODO: maybe this block is redundant
            if (boid.valBest < -100) {
                boid.valBest = -1;
                return boid.vArgBest;
            }
            else if (boid.valBest < 0)
                boid.valBest = val;


            if (boid.valCurrent > boid.valBest) {
                boid.vArgBest = boid.vArgCurrent;
                boid.valBest  = boid.valCurrent;
            }

            // Global Value Best    // TODO: maybe this block is redundant
            if (m_globalValBest < -100) {
                m_globalValBest = -1;
                return m_vGlobalArgBest;
            }
            else if (m_globalValBest < 0)
                m_globalValBest = val;


            if (boid.valCurrent > m_globalValBest) {
                m_vGlobalArgBest = boid.vArgBest;
                m_globalValBest = boid.valCurrent;
            }
        } // Boids


        // Update vVelocity and vParams of every boid
        for (Boid& boid : m_vBoids) {
            float r1 = random::U<float>();
            float r2 = random::U<float>();
            for (auto d = 0; d < m_vParams.size(); d++) {
                boid.vVelocity[d] = m_w * boid.vVelocity[d] + m_c1 * r1 * (boid.vArgBest[d] - boid.vArgCurrent[d]) + m_c2 * r2 * (m_vGlobalArgBest[d] - boid.vArgCurrent[d]);
                boid.vArgCurrent[d] += boid.vVelocity[d];	        // I think velocity is the same as deltas in Powell. Maybe re-use the corresponding container
            }
            boid.valCurrent = -101;
        }


        m_iteration++;
        // Printing out the information
        printf("[%zu]:\t", m_iteration);
        // for (float& param : m_vGlobalArgBest) printf("%.2f\t", param);
        printf("%.2f\n", m_globalValBest);

        return m_vGlobalArgBest;
	}

	// TODO: implement this function
	bool CParamEstimationPSO::isConverged(void) const {
        return m_iteration >= MAX_NR_ITERATIONS;
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
			
            
            // Fills boid.valCurrent, boid.valBest
            for (Boid& boid : m_vBoids) {
                
                // return boid.vParams
                // expect that the next call getParams(val): objectiveFunct_val = val = objectiveFunct(boid.vParams);

                boid.valCurrent = objectiveFunct(boid.vArgCurrent);	        // <= objectiveFunct_val will be achieved from an argument of the function
                
                boid.valBest = objectiveFunct(boid.vArgBest);				// <= boid.valBest will be achieved from an argument of the function
                      
                if (boid.valCurrent > boid.valBest) {
                    boid.vArgBest = boid.vArgCurrent;
                    boid.valBest = boid.valCurrent;
                }

                m_globalValBest = objectiveFunct(m_vGlobalArgBest);		       // <= gBest_val will be achieved from an argument of the function
                

                if (boid.valCurrent > m_globalValBest) {
                    m_vGlobalArgBest = boid.vArgBest;
                    m_globalValBest = -101;
                }
            }
            
			// Update vVelocity and vParams of every boid
			for (Boid& boid : m_vBoids) {
                float r1 = random::U<float>();
                float r2 = random::U<float>();
                for (auto d = 0; d < m_vParams.size(); d++) {
                    boid.vVelocity[d] = m_w * boid.vVelocity[d] + m_c1 * r1 * (boid.vArgBest[d] - boid.vArgCurrent[d]) + m_c2 * r2 * (m_vGlobalArgBest[d] - boid.vArgCurrent[d]);
                    boid.vArgCurrent[d] = boid.vArgCurrent[d] + boid.vVelocity[d];	    // I think velocity is the same as deltas in Powell. Maybe re-use the corresponding container
                }
                boid.valCurrent = -101;
            }
        
        
        
        
        }
		return m_vGlobalArgBest;
    }

}
