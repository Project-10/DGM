// Samples Accumulator class interface
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	// ================================ Samples Accumulator Class ==============================
	/**
	* @brief Samples accumulator class
	* @details This class allows for storing samples (\a e.g. n-dimensinal points in feature space) into the memory
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/		
	class CSamplesAccumulator
	{
	public:
		DllExport CSamplesAccumulator(void);
		DllExport ~CSamplesAccumulator(void);
		
		/**
		* @brief Adds new sample to the accumulator
		* @param sample The sample : Mat(size: n x 1; type: CV_64FC1). All the samples must have the same length n
		*/		
		DllExport void		addSample(const Mat &sample);
		/**
		* @brief Returns the sample with index \b idx
		* @param idx Sample index in storage
		* @return The Sample : Mat(size: n x 1; type: CV_64FC1)
		*/		
		DllExport Mat		getSample(size_t idx) const;
		/**
		* @brief Resets the accumulator 
		*/
		DllExport void		reset(void);
		/**
		* @brief Returns the number of all samples in accumulator
		* @return The number of samples
		*/		
		DllExport size_t	getNumSamples(void) const {return m_nSamples;}


	private:
		vec_mat_t	m_vClusters;			///< The set of all samples
		size_t		m_nSamples;				///< The number of samples 
		int			m_sampleLength;			///< Length of the sample vector (should be equal to nFeatures)

		const static size_t CLUSTER_SIZE = 1000;		///< The size of one cluster 
	};
}

