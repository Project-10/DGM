#pragma once
#include "types.h"
#include "macroses.h"
#include "Random.h"

namespace DirectGraphicalModels { namespace parallel {
// ------------------------------------------- GEMM ------------------------------------------
// --------------------- fast generalized matrix multiplication with PPL ---------------------
	namespace {
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
	}
	/**
	* @brief Fast generalized matrix multiplication.
	* @param A first multiplied input matrix that should have CV_32FC1, CV_64FC1, CV_32FC2, or CV_64FC2 type.
	* @param B second multiplied input matrix of the same type as src1.
	* @param alpha weight of the matrix product.
	* @param C third optional delta matrix added to the matrix product; it should have the same type as src1 and src2.
	* @param beta weight of src3.
	* @param res output matrix; it has the proper size and the same type as input matrices.
	*/
	DllExport inline void gemm(const Mat &A, const Mat &B, float alpha, const Mat &C, float beta, Mat &res)
	{
#ifdef ENABLE_PPL
		if (C.empty()) ppl_gemm(A, B, alpha, res);
		else ppl_gemm(A, B, alpha, C, beta, res);
#else
		cv::gemm(A, B, alpha, C, beta, res);
#endif
	}

	
	// ------------------------------------------- SUFFLE ------------------------------------------
	// ------------------------- fast random shuffle of elements with PPL  -------------------------
	namespace {
		inline void Swap(Mat &a, Mat &b)
		{
			add(a, b, a);		// a = a + b
			subtract(a, b, b);	// b = a - b
			subtract(a, b, a);	// a = a - b;
		}
	}

	/**
	* @brief Randomly shuffles the rows of the input matrix.
	* @details > This function supports PPL.
	* > When using PPL, the result of this function is biased.
	* @param[in,out] X The input/output data, which rows should be shffled.
	* @todo Eliminate the bias, caused by parallel processing.
	*/
	DllExport inline void shuffleRows(Mat &X)
	{
#ifdef ENABLE_PPL
		int nCores = MAX(1, std::thread::hardware_concurrency());
		int step = MAX(2, X.rows / (nCores * 10));
		concurrency::parallel_for(0, X.rows, step, [step, &X](int S) {
			int last = MIN(S + step, X.rows);
			for (int s = last - 1; s > S; s--) {				// s = [last - 1; S + 1]
				dword r = DirectGraphicalModels::random::u<dword>(S, s);			// r = [S; s] = [S; S + 1] -> [S; last - 1]
				if (r != s) Swap(X.row(s), X.row(r));
			}
		});
#else	
		for (int s = X.rows - 1; s > 0; s--) {			// s = [n-1; 1]
			int r = random::u<int>(0, s);		// r = [0; s] = [0; 1] -> [0; n-1]
			if (r != s)	Swap(X.row(s), X.row(r));
		}
#endif
	}

} }
