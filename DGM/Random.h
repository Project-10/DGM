// Random numbers generation class interface
// Written by Sergey G. Kosov in 2013 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	const int	NUM_SAMPLES = 12;			///< Number of samples for approximationg normal distribution
	static bool	isInitialized = false;
	// ================================ Random Class ==============================
	/**
	* @brief Random number generation class
	* @details This class allows to generate random numbers and vectors with different distributions
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CRandom
	{
	public:
		DllExport CRandom(void);
		DllExport ~CRandom(void) {};

		/**
		* @brief Returns an integer random number with uniform distribution (single precision)
		* @return An integer number in range between 0 and RAND_MAX 
		*/
		DllExport dword	u(void) const { return std::rand(); }
		/**
		* @brief Returns an integer random number with uniform distribution (double precision)
		* @return An integer number in range between 0 and sqr(RAND_MAX)
		*/
		DllExport dword	du(void) const;
		/**
		* @brief Returns a floating point random number with uniform distribution
		* @return A floating point number in range between 0 and 1 
		*/
		DllExport float	U(void)	const { return static_cast<float>(u()) / RAND_MAX; }
		/**
		* @brief Returns a floating point random number with uniform distribution
		* @param a The lower boundary
		* @param b The upper boundary
		* @return A floating point number in range between a and b 
		*/		
		DllExport float	U(float a, float b) const;	
		/**
		* @brief Returns a floating point random number with normal distribution
		* @return A floating point number with normal distribution with mean \f$\mu = 0\f$ and variance \f$\sigma = 1\f$
		*/		
		DllExport float	N(void) const;
		/**
		* @brief Returns a floating point random number with normal distribution
		* @param mu The mean of the distribution \f$\mu\f$
		* @param sigma The variance of the distribution \f$\sigma\f$
		* @return A floating point number with given normal distribution 
		*/		
		DllExport float	N(float mu, float sigma) const { return mu + sigma * N(); }
		/**
		* @brief Returns a vector of floating point random numbers with uniform distribution
		* @param k dimensions
		*@return A vector Mat(size: k x 1; type: CV_32FC1) of floating point numbers in range between 0 and 1 
		*/
		DllExport Mat	U(dword k) const; 
		/**
		* @brief Returns a vector of floating point random numbers with uniform distribution
		* @param k dimensions
		* @param a The lower boundary
		* @param b The upper boundary
		* @return A vector Mat(size: k x 1; type: CV_32FC1) of floating point numbers in range between a and b 
		*/	
		DllExport Mat	U(dword k, float a, float b) const; 
		/**
		* @brief Returns a vector of floating point random numbers with normal distribution
		* @param k dimensions
		* @return A vector Mat(size: k x 1; type: CV_32FC1) of floating point numbers with normal distribution with mean \f$\mu = 0\f$ and variance \f$\sigma = 1\f$
		*/
		DllExport Mat	N(dword k) const;
		/**
		* @brief Returns a vector of floating point random numbers with normal distribution
		* @param k dimensions
		* @param mu The mean of the distribution \f$\mu\f$
		* @param sigma The variance of the distribution \f$\sigma\f$
		* @return A vector Mat(size: k x 1; type: CV_32FC1) of floating point numbers with  given normal distribution 
		*/	
		DllExport Mat	N(dword k, float mu, float sigma) const;


	private:
		// Copy semantics are disabled
		CRandom(const CRandom &rhs) {}
		const CRandom & operator= (const CRandom & rhs) {return *this;}
	};
}

