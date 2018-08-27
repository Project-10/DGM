#pragma once

#include "types.h"

// ********************** CPairwisePotential **********************
class CEdgePotential
{
public:
	CEdgePotential(void) {}
	virtual ~CEdgePotential(void) {}
	virtual void apply(const Mat &src, Mat &dst, Mat &temp = EmptyMat) const = 0;


private:
	// Copy semantics are disabled
	CEdgePotential(const CEdgePotential &rhs) {}
	const CEdgePotential & operator= (const CEdgePotential & rhs) { return *this; }
};


// ********************** SemiMetricFunction **********************
class SemiMetricFunction {
public:
	// For two probabilities apply the semi metric transform: v_i = sum_j mu_ij u_j
	virtual void apply(const vec_float_t &src, vec_float_t &dst) const = 0;
};
