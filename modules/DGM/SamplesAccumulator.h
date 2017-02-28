// Samples Accumulator class interface
// Written by Sergey G. Kosov in 2015, 2017 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	// ================================ Samples Accumulator Class ==============================
	/**
	* @brief Samples accumulator abstract class
	* @details This class allows for storing samples (\a e.g. n-dimensinal points in feature space) into the memory
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CSamplesAccumulator
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* @param maxSamples Maximum number of samples to be used in training.
		* > Default value \b 0 means using all the samples.<br>
		* > If another value is specified, the class for training will use \b maxSamples random samples from the whole amount of samples, added via addSample() function
		*/
		CSamplesAccumulator(size_t maxSamples) : m_maxSamples(maxSamples ? maxSamples : std::numeric_limits<size_t>::max())	{ }
		virtual ~CSamplesAccumulator(void) {}

		/**
		* @brief Resets the accumulator
		*/
		virtual void	reset(void) = 0;
		/**
		* @brief Adds new sample to the accumulator
		* @param featureVector Multi-dimensinal point: Mat(size: nFeatures x 1)
		* @param state State (class) corresponding to the \b featureVector
		*/
		virtual void	addSample(const Mat &featureVector, byte state) = 0;


	protected:
		size_t		m_maxSamples;						// = INFINITY;				// for optimisation purposes

	private:
		// Copy semantics are disabled
		CSamplesAccumulator(const CSamplesAccumulator &rhs) {}
		const CSamplesAccumulator & operator= (const CSamplesAccumulator & rhs) { return *this; }
	};

	
	
	// ================================ Samples Accumulator Containers Class ==============================
	class CSamplesAccumulatorContainers : public CSamplesAccumulator
	{
	public:
		CSamplesAccumulatorContainers(byte nStates, size_t maxSamples = 0)
			: CSamplesAccumulator(maxSamples)
			, m_vSamplesAcc(vec_mat_t(nStates))
			, m_vNumInputSamples(vec_int_t(nStates, 0))
		{ }
		virtual ~CSamplesAccumulatorContainers(void) {}

		virtual void	reset(void);
		virtual void	addSample(const Mat &featureVector, byte state);

		/**
		* @brief Returns samples container for the state (class) \b state
		* @param state The state (class)
		* @return The container: Mat(size: nSamples x nFeatures)
		*/
		Mat		getSamplesContainer(byte state) const { return m_vSamplesAcc[state]; }
		/**
		* @brief Returns the number of stored samples in container for the state (class) \b state
		* @param state The state (class)
		* @return The number of samples
		*/
		int		getNumSamples(byte state) const;
		/**
		* @brief Returns the number of input samples in container for the state (class) \b state
		* @details This function retunts the number of samples added with the addSample() function.
		* Please note, that this number may be larger than the number of samples, actually stored in container,
		* that may me true, if the constructor's argument \b maxSamples was specified.
		* than the output of the getNumSamples()
		* @param state The state (class)
		* @return The number of samples
		*/
		int		getNumInputSamples(byte state) const;
		/**
		* @brief Releases memory of container for the state (class) \b state
		* @param state The state (class)
		*/
		void	release(byte state);


	private:
		vec_mat_t	m_vSamplesAcc;						// = vec_mat_t(nStates);	// Samples container for all states
		vec_int_t	m_vNumInputSamples;					// = vec_int_t(nStates, 0);	// Amount of input samples for all states
	};



	using vec_samplePair_t = std::vector<std::pair<Mat, byte>>;
	// ================================ Samples Accumulator Pairs Class ==============================
	class CSamplesAccumulatorPairs : public CSamplesAccumulator
	{
	public:
		CSamplesAccumulatorPairs(size_t maxSamples = 0) : CSamplesAccumulator(maxSamples) {}
		virtual ~CSamplesAccumulatorPairs(void) {}

		virtual void	reset(void);
		virtual void	addSample(const Mat &featureVector, byte state);


	private:
		vec_samplePair_t	m_vSamplesPair;
	};
}