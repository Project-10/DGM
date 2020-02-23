//
// Created by ahambasan on 22.02.20.
//

#include "PSO.h"
#include "macroses.h"

#include <iterator>
#include <vector>
#include <random>

DirectGraphicalModels::PSO::PSO() = default;

DirectGraphicalModels::PSO::PSO(const vec_float_t &vParams)
        : m_nParams(vParams.size()),
          m_vParams(vParams.size()),
          m_vMin(vParams.size()),
          m_vMax(vParams.size()),
          isThreadsEnabled(false) {

    for (size_t p = 0; p < vParams.size(); p++) {
        const float &param = vParams[p];
        m_vParams[p] = param;
    }
    m_nParams = vParams.size();

    // SOURCE: https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.00f, 10.00f);

    // initialize boids
    for (auto i = 0; i < NUMBER_BOIDS; i++) {
        Boid *b_x = new Boid;
        for (auto d = 0; d < m_nParams; d++) {
            b_x->pParams.push_back(dis(gen));
            b_x->velocity.push_back(1);
        }
        b_x->pBest = m_vParams;
        b_n.push_back(*b_x);
    }

    gBest = m_vParams;

    // initialize meta parameters
    c1 = C1_DEFAULT_VALUE;
    c2 = C2_DEFAULT_VALUE;
    w = W_DEFAULT_VALUE;

    reset();

}

void DirectGraphicalModels::PSO::reset() {
    std::fill(m_vMin.begin(), m_vMin.end(), -FLT_MAX);
    std::fill(m_vMax.begin(), m_vMax.end(), FLT_MAX);
    std::fill(m_vParams.begin(), m_vParams.end(), 0.0f);
}

void DirectGraphicalModels::PSO::setInitParams(const vec_float_t &vParams) {
    DGM_ASSERT_MSG(m_vParams.size() == vParams.size(),
                   "The size of the argument (%zu) does not correspond to the number of parameters (%zu)",
                   vParams.size(), m_nParams);

    for (size_t p = 0; p < vParams.size(); p++) {
        const float &param = vParams[p];
        if (param > m_vMax[p]) {
            DGM_WARNING("Argument[%zu]=%.2f exceeds the upper boundary %.2f and will not be set",
                        p, param, m_vMax[p]);
            continue;
        }
        if (param < m_vMin[p]) {
            DGM_WARNING("Argument[%zu]=%.2f exceeds the lower boundary %.2f and will not be set",
                        p, param, m_vMin[p]);
            continue;
        }
        m_vParams[p] = param;
    }
}

void DirectGraphicalModels::PSO::setMinParams(const vec_float_t &vMinParam) {
    DGM_ASSERT_MSG(m_vMin.size() == vMinParam.size(),
                   "The size of the argument (%zu) does not correspond to the number of parameters (%zu)",
                   vMinParam.size(), m_nParams);

    for (size_t p = 0; p < vMinParam.size(); p++) {
        const float &minParam = vMinParam[p];
        if (minParam > m_vParams[p])
            DGM_WARNING(
                    "Argument[%zu]=%.2f contradicts the parameter value %.2f and will not be set",
                    p, minParam, m_vParams[p]);
        else m_vMin[p] = minParam;
    }
}

void DirectGraphicalModels::PSO::setMaxParams(const vec_float_t &vMaxParam) {
    DGM_ASSERT_MSG(m_vMax.size() == vMaxParam.size(),
                   "The size of the argument (%zu) does not correspond to the number of parameters (%zu)",
                   vMaxParam.size(), m_nParams);

    for (size_t p = 0; p < vMaxParam.size(); p++) {
        const float &maxParam = vMaxParam[p];
        if (maxParam < m_vParams[p])
            DGM_WARNING(
                    "Argument[%zu]=%.2f contradicts the parameter value %.2f and will not be set",
                    p, maxParam, m_vParams[p]);
        else m_vMax[p] = maxParam;
    }
}

vec_float_t DirectGraphicalModels::PSO::getParams(float (*objectiveFunct)(vec_float_t)) {
    if (isThreadsEnabled) {
        std::vector<std::thread> threads_v;
        for (size_t i = 0; i < NUMBER_BOIDS; i++) {
            threads_v.emplace_back(&DirectGraphicalModels::PSO::runPSO_withThreads,
                    this, objectiveFunct, i);
        }
        std::for_each(threads_v.begin(), threads_v.end(), [](std::thread& th) {
            th.join();
        });

        return gBest;
    } else {
        runPSO(objectiveFunct);
        return gBest;
    }
}

void DirectGraphicalModels::PSO::runPSO(float (*objectiveFunct)(vec_float_t)) {
    size_t it = 0;
    while (it < MAX_NR_ITERATIONS) {
        for (auto i = 0; i < NUMBER_BOIDS; i++) {
            float objectiveFunct_val = objectiveFunct(b_n[i].pParams);
            float pBest_val = objectiveFunct(b_n[i].pBest);
            if (objectiveFunct_val < pBest_val) {
                b_n[i].pBest = b_n[i].pParams;
            }

            float gBest_val = objectiveFunct(gBest);
            if (objectiveFunct_val < gBest_val) {
                gBest = b_n[i].pBest;
            }
        }

        for (auto i = 0; i < NUMBER_BOIDS; i++) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0, 1);

            float r1 = dis(gen);
            float r2 = dis(gen);
            for (auto d = 0; d < m_nParams; d++) {
                b_n[i].velocity[d] = w * b_n[i].velocity[d] + c1 * r1 * (b_n[i].pBest[d] - b_n[i].pParams[d]) +
                                     c2 * r2 * (gBest[d] - b_n[i].pParams[d]);
                b_n[i].pParams[d] = b_n[i].pParams[d] + b_n[i].velocity[d];
            }
        }

        it++;
    }
}

void DirectGraphicalModels::PSO::runPSO_withThreads(float (*objectiveFunct)(vec_float_t), size_t idx) {
    size_t it = 0;
    while (it < MAX_NR_ITERATIONS) {
        float objectiveFunct_val = objectiveFunct(b_n[idx].pParams);
        float pBest_val = objectiveFunct(b_n[idx].pBest);
        if (objectiveFunct_val < pBest_val) {
            b_n[idx].pBest = b_n[idx].pParams;
        }

        mtx.lock();
        float gBest_val = objectiveFunct(gBest);
        if (objectiveFunct_val < gBest_val) {
            gBest = b_n[idx].pBest;
        }
        mtx.unlock();


        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);

        float r1 = dis(gen);
        float r2 = dis(gen);
        for (auto d = 0; d < m_nParams; d++) {
            b_n[idx].velocity[d] = w * b_n[idx].velocity[d] + c1 * r1 * (b_n[idx].pBest[d] - b_n[idx].pParams[d]) +
                                 c2 * r2 * (gBest[d] - b_n[idx].pParams[d]);
            b_n[idx].pParams[d] = b_n[idx].pParams[d] + b_n[idx].velocity[d];
        }

        it++;
    }
}

bool DirectGraphicalModels::PSO::isMultiThreadingEnabled() {
    return this->isThreadsEnabled;
}

void DirectGraphicalModels::PSO::enableMultiThreading() {
    this->isThreadsEnabled = true;
}

void DirectGraphicalModels::PSO::enableMultiThreading(bool enable) {
    this->isThreadsEnabled = enable;
}

void DirectGraphicalModels::PSO::disableMultiThreading() {
    this->isThreadsEnabled = false;
}

bool DirectGraphicalModels::PSO::isConverged() const {
    for (const bool& converged : m_vConverged) if (!converged) return false;
    return true;
}

DirectGraphicalModels::PSO::~PSO() = default;