#pragma once

#include "types.h"

// This function defines a simplified interface to the permutohedral lattice
// We assume a filter standard deviation of 1
class CPermutohedral;

class CFilter {
public:
	// Use different source and target features
	CFilter(const Mat &src_features, const Mat &dst_features);
	// Use the same source and target features
	CFilter(const Mat &features);
	~CFilter(void);
	
    // Filter a bunch of values
    void filter(const Mat &src, Mat &dst);


private:
	int m_n1;
	int m_o1;
	int m_n2;
	int m_o2;
	
	CPermutohedral * m_pPermutohedral;


private:
    // Copy semantics are disabled
    CFilter(const CFilter &rhs) {}
    const CFilter & operator= (const CFilter & rhs) { return *this; }
};
