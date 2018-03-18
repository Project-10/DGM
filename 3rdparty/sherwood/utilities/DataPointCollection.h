#pragma once

#include <vector>

#include "../Sherwood.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
/**
@brief A collection of data points, each represented by a float[] and (optionally) associated with a string class label and/or a float target value.
*/
	class DataPointCollection: public IDataPointCollection
	{
	public:
/**
@brief Count the data points in this collection.
@return The number of data points
*/
		size_t Count(void) const { return m_vData.size() / m_dimension;}

/**
@brief Count the data points with state (class) \a state in this collection.
@param state The state (class)
@return The number of data points
*/
		size_t Count(unsigned char state) const;

/**
@brief Get the specified data point.
@param i Zero-based data point index.
@return Pointer to the first element of the data point.
*/
		const float* GetDataPoint(int i) const {return &m_vData[i * m_dimension];}

/**
@brief Get the class label for the specified data point (or raise an exception if these data points do not have associated labels).
@param i Zero-based data point index.
@return A zero-based integer class label.
*/
		int GetIntegerLabel(int i) const;


	public:
		int							m_dimension;	///< The dimension of the feature vector
		std::vector<float>			m_vData;		///< Data container for the feature values
		std::vector<unsigned char>	m_vLabels;		///< Data container for correspinding sate (class) values
	};
} } }
