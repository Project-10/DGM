#include "TrainEdgePotts.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
    Mat CTrainEdgePotts::calculateEdgePotentials(const Mat &, const Mat &, const vec_float_t &vParams) const
    {
		if (vParams.size() == 1)				return getDefaultEdgePotentials(vParams[0], m_nStates);
		else if (vParams.size() == m_nStates)	return getDefaultEdgePotentials(vParams);
		else DGM_ASSERT_MSG(false, "Wrong number of parameters: %zu. It must be either %d or %u", vParams.size(), 1, m_nStates);
    }
}
