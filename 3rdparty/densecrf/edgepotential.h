#pragma once

#include "types.h"

// ********************** CPairwisePotential **********************
class CEdgePotential
{
public:
    CEdgePotential(void) = default;
    CEdgePotential(const CEdgePotential &rhs) = delete;
    virtual ~CEdgePotential(void) = default;
    const CEdgePotential & operator= (const CEdgePotential & rhs) = delete;
    
	virtual void apply(const Mat &src, Mat &dst, Mat &temp = EmptyMat) const = 0;
};
