//
// Created by ahambasan on 22.02.20.
// Impelemented by sabyrrakhim06 on 19.04.21.
//

#include "ParamEstimationPSO.h"
#include "random.h"

namespace DirectGraphicalModels {
    CParamEstimationPSO::CParamEstimationPSO(size_t nParams)
            : CParamEstimation(nParams), m_vBoids(NUMBER_BOIDS) {
        reset();
    }

    void CParamEstimationPSO::reset() {
        // initialize meta parameters
        m_c1 = C1_DEFAULT_VALUE;
        m_c2 = C2_DEFAULT_VALUE;
        m_w = W_DEFAULT_VALUE;

        std::fill(m_vMin.begin(), m_vMin.end(), -FLT_MAX);
        std::fill(m_vMax.begin(), m_vMax.end(), FLT_MAX);
        std::fill(m_vParams.begin(), m_vParams.end(), 0.0f);

        // initialize boids
        for (Boid &boid : m_vBoids) {
            boid.vArgCurrent.resize(m_vParams.size());
            for (float &p : boid.vArgCurrent)
                p = random::U(-100.0f, 100.0f);
            boid.valCurrent = std::make_pair(UNINITIALIZED, true);

            boid.vArgBest = m_vParams;        // m_vParams is empty. May be an error here.
            boid.valBest = UNINITIALIZED;

            boid.vVelocity.resize(m_vParams.size());
            std::fill(boid.vVelocity.begin(), boid.vVelocity.end(), 0.0);
        }
        m_globalValBest = UNINITIALIZED;
    }

    vec_float_t CParamEstimationPSO::getParams(float val) {
        // On the first call we get the value for the m_globalValBest and boid.vArgBest 
        if (m_globalValBest == UNINITIALIZED) {
            m_globalValBest = val;
            for (Boid &boid : m_vBoids)
                boid.valBest = val;
        }

        for (Boid &boid : m_vBoids) {

            // Boid: Value Current
            if (boid.valCurrent.first == UNINITIALIZED && boid.valCurrent.second) {
                boid.valCurrent.second = false;
                return boid.vArgCurrent;
            } else if (!boid.valCurrent.second)
                boid.valCurrent = std::make_pair(val, true);

            if (boid.valCurrent.first > boid.valBest) {
                boid.vArgBest = boid.vArgCurrent;
                boid.valBest = boid.valCurrent.first;
            } else if (fabs(boid.valCurrent.first - boid.valBest) < FLT_EPSILON) {
                float p = random::U<float>();
                if (p < 0.5) {
                    boid.vArgBest = boid.vArgCurrent;
                    boid.valBest = boid.valCurrent.first;
                }
            }

            if (boid.valBest > m_globalValBest) {
                m_vParams = boid.vArgBest;
                m_globalValBest = boid.valBest;
                boid.hasConverged = false;
            } else if (fabs(boid.valCurrent.first - m_globalValBest) < FLT_EPSILON) {
                boid.hasConverged = true;
            }
        } // Boids


        // Update vVelocity and vParams of every boid
        for (Boid& boid : m_vBoids) {
            for (auto d = 0; d < m_vParams.size(); d++) {
                // initialize random variables 
                float r1 = random::U<float>();
                float r2 = random::U<float>();
                boid.vVelocity[d] = m_w * boid.vVelocity[d] + m_c1 * r1 * (boid.vArgBest[d] - boid.vArgCurrent[d]) + m_c2 * r2 * (m_vGlobalArgBest[d] - boid.vArgCurrent[d]);
                boid.vArgCurrent[d] += boid.vVelocity[d];	        // I think velocity is the same as deltas in Powell. Maybe re-use the corresponding container
            }
            boid.valCurrent = std::make_pair(UNINITIALIZED, true);
        }

        return m_vParams;
    }

    bool CParamEstimationPSO::isConverged(void) const {
        for (auto boid : m_vBoids) {
            if (!boid.hasConverged)
                return false;
        }
        return true;
    }
}