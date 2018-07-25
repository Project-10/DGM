#pragma once

#include "types.h"

// This function defines a simplified interface to the permutohedral lattice
// We assume a filter standard deviation of 1
class Permutohedral;

class Filter {
public:
	// Use different source and target features
	Filter(const float * source_features, int N_source, const float * target_features, int N_target, int feature_dim);
	// Use the same source and target features
	Filter(const float * features, int N, int feature_dim);
	~Filter(void);
	// Filter a bunch of values
	void filter(const float * source, float * target, int value_size);


protected:
	int n1_;
	int o1_;
	int n2_;
	int o2_;
	
	Permutohedral * permutohedral_;

	// Don't copy
	Filter(const Filter& filter) {}
};