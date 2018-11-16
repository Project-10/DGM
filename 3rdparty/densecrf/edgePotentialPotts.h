#pragma once

#include "edgepotential.h"
#include "permutohedral.h"


// ********************** CPottsPotential **********************
class CEdgePotentialPotts : public CEdgePotential
{
public:
	CEdgePotentialPotts(const Mat &features, float weight = 1.0f, const std::function<void(const Mat &src, Mat &dst)> &SemiMetricFunction = {}, bool per_pixel_normalization = true);
	virtual ~CEdgePotentialPotts(void) {}

	virtual void apply(const Mat &pots, Mat &dst) const;


private:
	std::unique_ptr<CPermutohedral>					m_pLattice;
	float											m_weight;
	Mat												m_norm;
    std::function<void(const Mat &src, Mat &dst)>	m_function;
};

