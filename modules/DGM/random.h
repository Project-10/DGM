// Random numbers generation
// Written by Sergey G. Kosov in 2013 for Project X
// Simplified (C++11) by Sergey G. Kosov in 2016 for Project X
#pragma once

#include "types.h"
#include <random>

namespace DirectGraphicalModels
{
	// ================================ Random Namespace ==============================
	/**
	* @brief Random number generation
	* @details This namespace collects methods for generating random numbers and vectors with uniform and normal distributions
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	namespace random {
		/**
		* @brief Returns an integer random number with uniform distribution
		* @details This function produces random integer values \a i, uniformly distributed on the closed interval [\b min, \b max], that is, distributed according to the discrete probability function:
		* \f[ P(i\,|\,min,max)=\frac{1}{max-min+1}, min \leq i \leq max \f]
		* > This function is thread-safe
		* @tparam T An integer type: \a short, \a int, \a long, \a long \a long, \a unsigned \a short, \a unsigned \a int, \a unsigned \a long, or \a unsigned \a long \a long
		* @param min The lower boudaty of the interval
		* @param max The upper boundary of the interval
		* @returns The random number from interval [\b min, \b max]
		*/
		template <typename T>
		inline T u(T min, T max)
		{
			static thread_local std::mt19937 generator(static_cast<unsigned int>(clock() + std::hash<std::thread::id>()(std::this_thread::get_id())));
			std::uniform_int_distribution<T> distribution(min, max);
			return distribution(generator);
		}
		/**
		* @brief Returns a floating-point random number with uniform distribution
		* @details This function produces random floating-point values \a i, uniformly distributed on the interval [\b min, \b max), that is, distributed according to the probability function: 
		* \f[ P(i\,|\,min,max)=\frac{1}{max-min}, min \leq i < max \f] 
		* > This function is thread-safe
		* @tparam T A floating-point type: \a float, \a double, or \a long \a double
		* @param min The lower boudaty of the interval
		* @param max The upper boundary of the interval
		* @return The random number from interval [\b min, \b max)
		*/
		template <typename T>
		inline T U(T min = 0, T max = 1)
		{
			static thread_local std::mt19937 generator(static_cast<unsigned int>(clock() + std::hash<std::thread::id>()(std::this_thread::get_id())));
			std::uniform_real_distribution<T> distribution(min, max);
			return distribution(generator);
		}
		/**
		* @brief Returns a floating-point random number with normal distribution
		* @details This function generates random numbers according to the <a href="https://en.wikipedia.org/wiki/Normal_distribution">Normal (or Gaussian) random number distribution</a>:
		* \f[ f(x\,;\,\mu,\sigma)=\frac{1}{\sqrt{2\sigma^2\pi}} exp{\frac{-(x-\mu)^2}{2\sigma^2}} \f]
		* > This function is thread-safe
		* @tparam T A floating-point type: \a float, \a double, or \a long \a double
		* @param mu The <a href="https://en.wikipedia.org/wiki/Mean">mean</a> \f$\mu\f$
		* @param sigma The <a href="https://en.wikipedia.org/wiki/Standard_deviation">standard deviation</a> \f$\sigma\f$
		* @return A floating point number with normal distribution
		*/
		template <typename T>
		inline T N(T mu = 0, T sigma = 1)
		{
			static thread_local std::mt19937 generator(static_cast<unsigned int>(clock() + std::hash<std::thread::id>()(std::this_thread::get_id())));
			std::normal_distribution<T> distribution(mu, sigma);
			return distribution(generator);
		}


		/**
		* @brief Returns a matrix of floating-point random numbers with uniform distribution
		* @param size Size of the resulting matrix
		* @param type Type of the resulting matrix
		* @param min The lower boundary
		* @param max The upper boundary
		* @return A matrix of floating-point numbers in range between [\b min and \b max)
		*/
		inline Mat U(cv::Size size, int type, double min = 0, double max = 1)
		{
			static thread_local RNG rng(static_cast<unsigned int>(clock() + std::hash<std::thread::id>()(std::this_thread::get_id())));
			Mat res(size, type);
			rng.fill(res, RNG::UNIFORM, min, max);
			return res;
		}
		/**
		* @brief Returns a matrix of floating-point random numbers with normal distribution
		* @param size Size of the resulting matrix
		* @param type Type of the resulting matrix
		* @param mu The mean \f$\mu\f$
		* @param sigma The standard deviation \f$\sigma\f$
		* @return A matrix of floating-point numbers with normal distribution
		*/
		inline Mat N(cv::Size size, int type, double mu = 0, double sigma = 1)
		{
			static thread_local RNG rng(static_cast<unsigned int>(clock() + std::hash<std::thread::id>()(std::this_thread::get_id())));
			Mat res(size, type);
			rng.fill(res, RNG::NORMAL, mu, sigma);
			return res;
		}

	}
}

