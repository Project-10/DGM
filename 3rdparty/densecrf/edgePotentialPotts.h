#pragma once

#include "edgepotential.h"
#include "permutohedral.h"


// ********************** CPottsPotential **********************
class CEdgePotentialPotts : public CEdgePotential
{
public:
    CEdgePotentialPotts(const Mat &features, float w = 1.0f, const std::function<void(const vec_float_t &src, vec_float_t &dst)> &SemiMetricFunction = {}, bool per_pixel_normalization = true);
	virtual ~CEdgePotentialPotts(void) {}

	void apply(const Mat &src, Mat &dst, Mat &temp = EmptyMat) const;


	// TODO: try without members
private:
	float                           m_w;
	std::unique_ptr<CPermutohedral> m_pLattice;
	Mat                             m_norm;
    const std::function<void(const vec_float_t &src, vec_float_t &dst)> &m_function;
};

