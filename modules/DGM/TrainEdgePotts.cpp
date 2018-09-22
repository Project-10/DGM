#include "TrainEdgePotts.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
    Mat CTrainEdgePotts::calculateEdgePotentials(const Mat &, const Mat &, float *params, size_t params_len) const
    {
        DGM_ASSERT_MSG((params_len == 1) || (params_len == m_nStates), "Wrong number of parameters: %zu. It must be either %d or %u", params_len, 1, m_nStates);

        if (params_len == 1) return getDefaultEdgePotentials(params[0], m_nStates);
        else				 return getDefaultEdgePotentials(vec_float_t(params, params + m_nStates * sizeof(float)));
    }
}
