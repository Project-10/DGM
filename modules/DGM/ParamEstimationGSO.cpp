//
//  created by sabyrrakhim06 on 10.05.2021
//  inspired by PSO algorithm implemented by ahambasan and sabyrrakhim06
//

#include "ParamEstimationGSO.h"
#include "random.h"
 
namespace DirectGraphicalModels {
    CParamEstimationGSO::CParamEstimationGSO(size_t nParams)
        : CParamEstimation(nParams), m_vBoids(NUMBER_BOIDS), m_vGboids(NUMBER_SUBSWARMS) {
                reset();
    }
 
    void CParamEstimationGSO::reset() {
        // TODO: Divide swarm to subswarms
        // initialize meta parameters
        m_c1 = C1_DEFAULT_VALUE;
        m_c2 = C2_DEFAULT_VALUE;
 
        std::fill(m_vMin.begin(), m_vMin.end(), -FLT_MAX);
        std::fill(m_vMax.begin(), m_vMax.end(), FLT_MAX);
        std::fill(m_vParams.begin(), m_vParams.end(), 0.0f);
 
        // initialize boids
        for (Boid &boid : m_vBoids) {
            boid.vArgCurrent.resize(m_vParams.size());
            boid.vGlobalVal.resize(m_vParams.size());
            std::fill(boid.vGlobalVal.begin(), boid.vGlobalVal.end(), 0.0);
            for (float &p : boid.vArgCurrent)
                p = random::U(-100.0f, 100.0f);
            boid.valCurrent = std::make_pair(UNINITIALIZED, true);
 
            boid.vArgBest = m_vParams;
            boid.valBest = UNINITIALIZED;
 
            boid.vVelocity.resize(m_vParams.size());
            std::fill(boid.vVelocity.begin(), boid.vVelocity.end(), 0.0);
        }
        m_globalValBest = UNINITIALIZED;
        m_globalGalBest = UNINITIALIZED;
    }
 
    vec_float_t CParamEstimationGSO::getParams(float val) {
        // On the first call we get the value for the m_globalValBest and boid.vArgBest
        if (m_globalValBest == UNINITIALIZED) {
            m_globalValBest = val;
            for (Boid &boid : m_vBoids)
                boid.valBest = val;
        }
 
        // Galactic Swarm Optimization
        for(auto epoch = 1; epoch < EPOCH_MAX; epoch++) {
            // PSO Level 1
            for(auto i = 0; i < NUMBER_SUBSWARMS; i++) {
                for(auto k = 0; k < L1; k++) {
                    m_w1 = 1 - k / (L1 + 1);
                    for(Boid &boid : m_vBoids) {
                        float r1 = random::U<float>();
                        float r2 = random::U<float>();
                        boid.vVelocity[i] = m_w1 * boid.vVelocity[i] + m_c1 * r1 * (boid.vArgBest[i] - boid.vArgCurrent[i]) + m_c2 * r2 * (m_vParams[i] - boid.vArgCurrent[i]);
                        boid.vArgCurrent[i] += boid.vVelocity[i];
 
                        if(boid.valCurrent.first < boid.valBest) {
                            boid.vArgBest = boid.vArgCurrent;
                            boid.valBest = boid.valCurrent.first;
 
                            if(boid.valBest < m_globalValBest) {
                                m_vParams = boid.vArgBest;
                                m_globalValBest = boid.valBest;
 
                                if(m_globalValBest < m_globalGalBest) {
                                    m_vParams = boid.vGlobalVal;
                                    m_globalGalBest = m_globalValBest;
                                }
                            }
                        }
                    }
                }
            }
            // TODO: Store optimum from each subswarm
            // PSO Level 2
            // initialize meta parameters
            m_c3 = C3_DEFAULT_VALUE;
            m_c4 = C4_DEFAULT_VALUE;
 
            // initialize superswarm
            for (GBoid &gboid : m_vGboids) {
                gboid.vGalCurrent.resize(m_vParams.size());
                gboid.galCurrent.first = m_globalValBest;
                gboid.vGalCurrent = m_vParams;
                //std::fill(gboid.vGalCurrent.begin(), gboid.vGalCurrent.end(), 0.0f);
 
 
                gboid.vGalBest.resize(m_vParams.size());
                std::fill(gboid.vGalBest.begin(), gboid.vGalBest.end(), 0.0f);
 
                gboid.galBest = UNINITIALIZED;
 
                gboid.vGVelocity.resize(m_vParams.size());
                std::fill(gboid.vGVelocity.begin(), gboid.vGVelocity.end(), 0.0f);
            }
 
 
            for(auto k = 0; k < L2; k++) {
                m_w2 = 1 - k / (L2 + 1);
                for(GBoid &gboid : m_vGboids) {
                    for(auto d = 0; d < m_vParams.size(); d++) {
                        float r3 = random::U<float>();
                        float r4 = random::U<float>();
                        gboid.vGVelocity[d] = m_w2 * gboid.vGVelocity[d] + m_c3 * r3 * (gboid.vGalBest[d] - gboid.vGalCurrent[d]) + m_c4 * r4 * (m_vParams[d] - gboid.vGalCurrent[d]);
                        gboid.vGalCurrent[d] += gboid.vGVelocity[d];
                    }
                    if(gboid.galCurrent.first > gboid.galBest) {
                        gboid.vGalBest = gboid.vGalCurrent;
                        gboid.galBest = gboid.galCurrent.first;
                        if(gboid.galBest > m_globalGalBest) {
                            m_vParams = gboid.vGalBest;
                            m_globalGalBest = gboid.galBest;
                        }
                    }
                }
            }
        }
 
        return m_vParams;
    }
 
    bool CParamEstimationGSO::isConverged(void) const {
        static int n = 0;
        n++;
        return n > 1000;
    }
 
}
