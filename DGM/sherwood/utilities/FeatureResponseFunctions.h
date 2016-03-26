// This file defines some IFeatureResponse implementations used by the example code in
// Classification.h, DensityEstimation.h, etc. Note we represent IFeatureResponse
// instances using simple structs so that all tree data can be stored
// contiguously in a linear array.
#pragma once

#include <string>

#include "..\Sherwood.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
	class Random;

// =========================== Linear Feature Response Class ===========================
/** 
@brief A feature that orders data points using a linear combination of their coordinates, i.e. by projecting them onto a given direction vector.
*/		
	class LinearFeatureResponse
	{
	public:
/**@brief Default constuctor*/		
		LinearFeatureResponse(void) : m_nFeatures(0) {} 
/**
@brief Create a LinearFeatureResponse instance for the specified direction vector.
@param nFeatures Number of features
@param pDx The array of elements of the direction vector
*/
		LinearFeatureResponse(unsigned char nFeatures, float *pDx);
		~LinearFeatureResponse(void) {};

/**
@brief Create a LinearFeatureResponse2d instance with a random direction vector.
@param nFeatures Number of features
@param random A random number generator.
@returns A new LinearFeatureResponse2d instance.
*/
		static LinearFeatureResponse	CreateRandom(unsigned char nFeatures, Random& random);

/**
@brief Computes the response for the specified data point. 
@details IFeatureResponse implementation
@param data The data. 
@param index The index of the data point to be evaluated. 
@returns A single precision response value.
*/
		float							GetResponse(const IDataPointCollection& data, size_t index) const;


	private:
		float			m_pDx[256];
		unsigned char	m_nFeatures;
	};

} } }
