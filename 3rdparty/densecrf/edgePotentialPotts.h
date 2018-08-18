#pragma once

#include "edgepotential.h"
#include "permutohedral.h"


// ********************** CPottsPotential **********************
class CEdgePotentialPotts : public CEdgePotential
{
public:
	CEdgePotentialPotts(const float *pFeatures, word nFeatures, size_t nNodes, float w = 1.0f, const SemiMetricFunction *pFunction = NULL, bool per_pixel_normalization = true);
	virtual ~CEdgePotentialPotts(void) {}

	void apply(Mat &out, const Mat &in, Mat &temp) const;


	// TODO: try private
	// TODO: try without members
protected:
	size_t                          m_nNodes;
	float                           m_w;
	std::unique_ptr<CPermutohedral> m_pLattice;
	Mat                             m_norm;
    const SemiMetricFunction      * m_pFunction;
};

