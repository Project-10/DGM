//
// Created by ahambasan on 26.02.20.
//

#include "ParamEstAlgorithm.h"
#include "macroses.h"

bool DirectGraphicalModels::ParamEstAlgorithm::isConverged(void) const {
    for (const bool& converged : m_vConverged) if (!converged) return false;
    return true;
}

void DirectGraphicalModels::ParamEstAlgorithm::setMaxParams(const vec_float_t &vMaxParam) {
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

void DirectGraphicalModels::ParamEstAlgorithm::setMinParams(const vec_float_t &vMinParam) {
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

void DirectGraphicalModels::ParamEstAlgorithm::setInitParams(const vec_float_t &vParams) {
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

DirectGraphicalModels::ParamEstAlgorithm::ParamEstAlgorithm(size_t nParams)
        :  m_nParams(nParams), m_vParams(nParams), m_vMin(nParams), m_vMax(nParams), m_vConverged(nParams) {}

