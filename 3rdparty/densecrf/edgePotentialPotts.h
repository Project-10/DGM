#pragma once

#include "edgepotential.h"
#include "permutohedral.h"


// ********************** CPottsPotential **********************
class CEdgePotentialPotts : public CEdgePotential
{
public:
	CEdgePotentialPotts(const float *pFeatures, word nFeatures, size_t nNodes, float w, bool per_pixel_normalization = true);
	virtual ~CEdgePotentialPotts(void) {}

	void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const;


	// TODO: try private
	// TODO: try without members
protected:
	size_t                          m_nNodes;
	float                           m_w;
	std::unique_ptr<CPermutohedral> m_pLattice;
	vec_float_t                     m_vNorm;
};


// ********************** CSemiMetricPotential **********************
class CEdgePotentialPottsSemiMetric : public CEdgePotentialPotts
{
public:
	CEdgePotentialPottsSemiMetric(const float *features, word nFeatures, size_t nNodes, float w, const SemiMetricFunction *function, bool per_pixel_normalization = true)
		: CEdgePotentialPotts(features, nFeatures, nNodes, w, per_pixel_normalization)
		, m_pFunction(function)
	{ }
	virtual ~CEdgePotentialPottsSemiMetric(void) {}

	void apply(vec_float_t &out_values, const vec_float_t &in_values, vec_float_t &tmp, int value_size) const;


protected:
	const SemiMetricFunction * m_pFunction;
};