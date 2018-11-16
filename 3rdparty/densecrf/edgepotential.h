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
    
	/**
	* @brief
	* @param pots The node potentials: Mat(size: nNodes x nStates; type: CV_32FC1)
	* @param dst
	*/
	virtual void apply(const Mat &pots, Mat &dst) const = 0;
};
