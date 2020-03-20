//
// Created by ahambasan on 26.02.20.
//
#include "ParamEstAlgorithm.h"
#include "macroses.h"

namespace DirectGraphicalModels 
{
    CParamEstAlgorithm::CParamEstAlgorithm(size_t nParams)
        : m_vParams(nParams)
        , m_vDeltas(nParams)
        , m_vMin(nParams)
        , m_vMax(nParams)
    {}

    void CParamEstAlgorithm::setInitParams(const vec_float_t& vParams) {
        DGM_ASSERT_MSG(vParams.size() == m_vParams.size(),
            "The size of the argument (%zu) does not correspond to the number of parameters (%zu)",
            vParams.size(), m_vParams.size());

        for (size_t p = 0; p < vParams.size(); p++) {
            const float& param = vParams[p];
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

    void CParamEstAlgorithm::setDeltas(const vec_float_t& vDeltas) {
        DGM_ASSERT_MSG(vDeltas.size() == m_vDeltas.size(),
            "The size of the argument (%zu) ddoes not correspond to the number of parameters (%zu)",
            vDeltas.size(), m_vDeltas.size());

        m_vDeltas = vDeltas;
    }

    void CParamEstAlgorithm::setMinParams(const vec_float_t& vMinParam) {
        DGM_ASSERT_MSG(vMinParam.size() == m_vMin.size(),
            "The size of the argument (%zu) does not correspond to the number of parameters (%zu)",
            vMinParam.size(), m_vMin.size());

        for (size_t p = 0; p < vMinParam.size(); p++) {
            const float& minParam = vMinParam[p];
            if (minParam > m_vParams[p])
                DGM_WARNING(
                    "Argument[%zu]=%.2f contradicts the parameter value %.2f and will not be set",
                    p, minParam, m_vParams[p]);
            else m_vMin[p] = minParam;
        }
    }

    void CParamEstAlgorithm::setMaxParams(const vec_float_t& vMaxParam) {
        DGM_ASSERT_MSG(vMaxParam.size() == m_vMax.size(),
            "The size of the argument (%zu) does not correspond to the number of parameters (%zu)",
            vMaxParam.size(), m_vMax.size());

        for (size_t p = 0; p < vMaxParam.size(); p++) {
            const float& maxParam = vMaxParam[p];
            if (maxParam < m_vParams[p])
                DGM_WARNING(
                    "Argument[%zu]=%.2f contradicts the parameter value %.2f and will not be set",
                    p, maxParam, m_vParams[p]);
            else m_vMax[p] = maxParam;
        }
    }
}