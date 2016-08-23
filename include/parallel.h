#pragma once
#include "types.h"
#include "macroses.h"

#include <random>

namespace DirectGraphicalModels { namespace parallel {
// ------------------------------------------- GEMM ------------------------------------------
// --------------------- fast generalized matrix multiplication with PPL ---------------------
#ifdef ENABLE_PPL
	inline void ppl_gemm(const Mat &A, const Mat &B, float alpha, Mat &res)
	{
		DGM_ASSERT(A.cols == B.rows);
		if (res.empty()) res = Mat(A.rows, B.cols, CV_32FC1);
		DGM_ASSERT(res.rows == A.rows);
		DGM_ASSERT(res.cols == B.cols);

		const Mat _B = B.t();
		concurrency::parallel_for(0, res.rows, [&](int y) {
			float * pRes = res.ptr<float>(y);
			const float * pA = A.ptr<float>(y);
			for (register int x = 0; x < res.cols; x++) {
				const float * pB = _B.ptr<float>(x);
				float sum = 0.0f;
				for (register int k = 0; k < A.cols; k++)
					sum += pA[k] * pB[k];
				pRes[x] = alpha * sum;
			}
		});
	}

	inline void ppl_gemm(const Mat &A, const Mat &B, float alpha, const Mat &C, float beta, Mat &res)
	{
		DGM_ASSERT(A.cols == B.rows);
		if (res.empty()) res = Mat(A.rows, B.cols, CV_32FC1);
		DGM_ASSERT(res.rows == A.rows && res.rows == C.rows);
		DGM_ASSERT(res.cols == B.cols && res.cols == C.cols);

		const Mat _B = B.t();
		concurrency::parallel_for(0, res.rows, [&](int y) {
			float * pRes = res.ptr<float>(y);
			const float * pA = A.ptr<float>(y);
			const float * pC = C.ptr<float>(y);
			for (register int x = 0; x < res.cols; x++) {
				const float * pB = _B.ptr<float>(x);
				float sum = 0.0f;
				for (register int k = 0; k < A.cols; k++)
					sum += pA[k] * pB[k];
				pRes[x] = alpha * sum + beta * pC[x];
			}
		});
	}
#endif 

	inline void gemm(const Mat &A, const Mat &B, float alpha, const Mat &C, float beta, Mat &res)
	{
#ifdef ENABLE_PPL
		if (C.empty()) ppl_gemm(A, B, alpha, res);
		else ppl_gemm(A, B, alpha, C, beta, res);
#else
		gemm(A, B, alpha, C, beta, res);
#endif
	}

	/**
	* @brief Returns an integer random number with uniform distribution
	* @details This function produces random integer values \a i, uniformly distributed on the closed interval [\b min, \b max], that is, distributed according to the discrete probability function
	* \f[ P(i\,|\,min,max)=\frac{1}{max-min+1}, min \leq i \leq max \f]
	* @param min The lower boudaty of the interval
	* @param max The upper boundary of the interval
	* @returns The random number from interval [\b min, \b max]
	*/
	template <typename T>
	inline T rand(T min, T max)
	{
		static thread_local std::mt19937 generator(static_cast<unsigned int>(clock() + std::hash<std::thread::id>()(std::this_thread::get_id())));
		std::uniform_int_distribution<T> distribution(min, max);
		return distribution(generator);
	}


} }
