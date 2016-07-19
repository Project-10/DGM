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
	* @brief A feature that orders data points using a linear combination of their coordinates, i.e. by projecting them onto a given direction vector.
	*/		
	class LinearFeatureResponse
	{
	public:
		/**
		* @brief Default constuctor
		*/		
		LinearFeatureResponse(void);
		/**
		* @brief Copy constructor
		*/
		LinearFeatureResponse(const LinearFeatureResponse &copy);
		/**
		* @brief Create a LinearFeatureResponse instance for the specified direction vector.
		* @param nFeatures Number of features
		* @param pDx The array of elements of the direction vector
		*/
		LinearFeatureResponse(unsigned short nFeatures, float * pDx);
		LinearFeatureResponse & operator=(const LinearFeatureResponse &rhs);
		~LinearFeatureResponse(void);

		/**
		* @brief Create a LinearFeatureResponse instance with a random direction vector.
		* @param nFeatures Number of features
		* @param random A random number generator.
		* @returns A new LinearFeatureResponse instance.
		*/
		static LinearFeatureResponse	CreateRandom(unsigned short nFeatures, Random &random);
		
		/**
		* @brief Computes the response for the specified data point. 
		* @details IFeatureResponse implementation
		* @param data The data. 
		* @param index The index of the data point to be evaluated. 
		* @returns A single precision response value.
		*/
		float							GetResponse(const IDataPointCollection &data, size_t index) const;


	private:
		float			* m_pDx;
		unsigned short	  m_nFeatures;
	};

} } }
