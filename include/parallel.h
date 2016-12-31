#pragma once
#include "types.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace parallel {
#ifdef USE_PPL
// ------------------------------------------- GEMM ------------------------------------------
// --------------------- fast generalized matrix multiplication with PPL ---------------------
inline void ppl_gemm(const Mat &A, const Mat &B, float alpha, Mat &res)
		{
			DGM_ASSERT(A.cols == B.rows);
			if (res.empty()) res = Mat(A.rows, B.cols, CV_32FC1);
			DGM_ASSERT(res.rows == A.rows);
			DGM_ASSERT(res.cols == B.cols);

			const Mat _B = Mat(B.t());								// may be more stable under CPU full load
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

			const Mat _B = Mat(B.t());								// may be more stable under CPU full load
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
#ifdef USE_PPL
			if (C.empty()) ppl_gemm(A, B, alpha, res);
			else ppl_gemm(A, B, alpha, C, beta, res);
#else
			gemm(A, B, alpha, C, beta, res);
#endif
		}
} }
