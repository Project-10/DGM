// Written by Sergey Kosov in 2016 - 2017 for Project X
#pragma once

#include "types.h"
#include "macroses.h"
#include "Random.h"

namespace DirectGraphicalModels { namespace parallel {
// ------------------------------------------- GEMM ------------------------------------------
// --------------------- fast generalized matrix multiplication with PPL ---------------------
	namespace {
#ifdef ENABLE_AMP
		inline void amp_gemm(const Mat &A, const Mat &B, float alpha, Mat &res) 
		{
			DGM_ASSERT(A.cols == B.rows);
			if (res.empty()) res = Mat(A.rows, B.cols, CV_32FC1);
			DGM_ASSERT(res.rows == A.rows);
			DGM_ASSERT(res.cols == B.cols);

			concurrency::array_view<const float, 2> a(A.rows, A.cols, reinterpret_cast<float * const>(A.data));
			concurrency::array_view<const float, 2> b(B.rows, B.cols, reinterpret_cast<float * const>(B.data));
			concurrency::array_view<float, 2>       r(res.rows, res.cols, reinterpret_cast<float *> (res.data));
			concurrency::parallel_for_each(r.extent, [=](concurrency::index<2> idx) restrict(amp) {
				int y = idx[0];
				int x = idx[1];
				float sum = 0.0f;
				for (register int k = 0; k < a.extent[1]; k++) 
					sum += a(y, k) * b(k, x);
				r[idx] = alpha * sum;
			});
			r.synchronize();
		}

		inline void amp_gemm(const Mat &A, const Mat &B, float alpha, const Mat &C, float beta, Mat &res)
		{
			DGM_ASSERT(A.cols == B.rows);
			if (res.empty()) res = Mat(A.rows, B.cols, CV_32FC1);
			DGM_ASSERT(res.rows == A.rows && res.rows == C.rows);
			DGM_ASSERT(res.cols == B.cols && res.cols == C.cols);

			concurrency::array_view<const float, 2> a(A.rows, A.cols, reinterpret_cast<float * const>(A.data));
			concurrency::array_view<const float, 2> b(B.rows, B.cols, reinterpret_cast<float * const>(B.data));
			concurrency::array_view<const float, 2> c(C.rows, C.cols, reinterpret_cast<float * const>(C.data));
			concurrency::array_view<float, 2>       r(res.rows, res.cols, reinterpret_cast<float *> (res.data));
			concurrency::parallel_for_each(r.extent, [=](concurrency::index<2> idx) restrict(amp) {
				int y = idx[0];
				int x = idx[1];
				float sum = 0.0f;
				for (register int k = 0; k < a.extent[1]; k++)
					sum += a(y, k) * b(k, x);
				r[idx] = alpha * sum + beta * c[idx];
			});
			r.synchronize();
		}
#endif
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
#ifdef ENABLE_AMP
		if (C.empty()) amp_gemm(A, B, alpha, res);
		else amp_gemm(A, B, alpha, C, beta, res);
#else 
#ifdef ENABLE_PPL
		if (C.empty()) ppl_gemm(A, B, alpha, res);
		else ppl_gemm(A, B, alpha, C, beta, res);
#else
		cv::gemm(A, B, alpha, C, beta, res);
#endif
#endif
	}

	
	// -------------------------------------------- SORT -------------------------------------------
	// --------------------------- fast sorting of Mat elements with PPL  --------------------------
	namespace {
		inline void Swap(Mat &a, Mat &b, Mat &tmp = Mat())
		{
			a.copyTo(tmp);
			b.copyTo(a);
			tmp.copyTo(b);
		}

		template <typename T>
		inline void insertion_sort(Mat &m, int x, int begin, int end)
		{
			Mat tmp;
			for (int i = begin; i <= end; i++) {
				int j = i;
				while (j > begin && m.at<T>(j, x) < m.at<T>(j - 1, x)) {
					Swap(m.row(j), m.row(j - 1), tmp);
					j--;
				}
			}
		}
	
		template <typename T>
		inline void sequential_quick_sort(Mat &m, int x, int begin, int end, int threshold)
		{
			if (end - begin < threshold) insertion_sort<T>(m, x, begin, end);
			else {
				int	_begin	= begin;
				int	_end	= end;
				T	pivot	= m.at<T>((begin + end) / 2, x);

				// partition
				while (_begin <= _end) {
					while (m.at<T>(_begin, x) < pivot) _begin++;
					while (m.at<T>(_end,   x) > pivot) _end--;
					if (_begin <= _end) {
						Swap(m.row(_begin), m.row(_end));
						_begin++;
						_end--;
					}
				};

				// recursion 
				if (begin < _end)	sequential_quick_sort<T>(m, x, begin, _end, threshold);
				if (_begin < end)	sequential_quick_sort<T>(m, x, _begin, end, threshold);
			}
		}

#ifdef ENABLE_PPL
		template <typename T>
		inline void parallel_quick_sort(Mat &m, int x, int begin, int end, int threshold, int depthRemaining)
		{
			if (end - begin < threshold) insertion_sort<T>(m, x, begin, end);
			else {
				int	_begin	= begin;
				int	_end	= end;
				T	pivot	= m.at<T>((begin + end) / 2, x);

				// partition
				while (_begin <= _end) {
					while (m.at<T>(_begin, x) < pivot) _begin++;
					while (m.at<T>(_end,   x) > pivot) _end--;
					if (_begin <= _end) {
						Swap(m.row(_begin), m.row(_end));
						_begin++;
						_end--;
					}
				};

				// recursion 
				if (depthRemaining > 0)
					concurrency::parallel_invoke(
						[&, x, begin, _end] { if (begin < _end)	parallel_quick_sort<T>(m, x, begin, _end, threshold, depthRemaining - 1); },
						[&, x, end, _begin] { if (_begin < end)	parallel_quick_sort<T>(m, x, _begin, end, threshold, depthRemaining - 1); }
				);
				else {
					if (begin < _end)	sequential_quick_sort<T>(m, x, begin, _end, threshold);
					if (_begin < end)	sequential_quick_sort<T>(m, x, _begin, end, threshold);
				}
			}
		}
#endif
	}

	/**
	* @brief Sorts the rows of the input matrix by the given dimension.
	* @details The result of the sorting may is expressed as: \f$ m_{x,y} < m_{x,y+1}, \forall y \f$.
	* > This function supports PPL.
	* @tparam T The type of elements in matrix.
	* @param[in, out] m The input/output data, which rows should be sorted.
	* @param x The dimension along which the matrix is sorted.
	*/
	template <typename T>
	DllExport inline void sortRows(Mat &m, int x)
	{
		DGM_ASSERT(x < m.cols);
#ifdef ENABLE_PPL
		const int nCores = MAX(1, concurrency::CurrentScheduler::Get()->GetNumberOfVirtualProcessors());
		parallel_quick_sort<T>(m, x, 0, m.rows - 1, 200, static_cast<int>(log2f(float(nCores))) + 4);
#else 
		sequential_quick_sort<T>(m, x, 0, m.rows - 1, 200);
#endif
	}

	namespace {
		template <typename T>
		inline void deepSort(Mat &m, int depth, int begin, int end)
		{
			if (depth == m.cols) return;				// we are too deep
			if (begin == end)    return;				// do not sort one element

#ifdef ENABLE_PPL
			const int nCores = MAX(1, concurrency::CurrentScheduler::Get()->GetNumberOfVirtualProcessors());
			parallel_quick_sort<T>(m, depth, begin, end, 200, static_cast<int>(log2f(float(nCores))) + 4);
#else 
			sequential_quick_sort<T>(m, depth, begin, end, 200);
#endif

			int ref_pos = begin;
			T   ref_val = m.at<T>(begin, depth);
			for (int y = begin + 1; y <= end; y++) {
				T val = m.at<T>(y, depth);
				if (val != ref_val) {
					deepSort<T>(m, depth + 1, ref_pos, y - 1);
					ref_pos = y;
					ref_val = val;
				}
			} // y
		}
	}

	/**
	* @brief Sorts the rows of the input matrix
	* @details 
	* > This function supports PPL.
	* @tparam T The type of elements in matrix.
	* @param[in, out] m The input/output data, which rows should be sorted.
	*/
	template <typename T>
	DllExport inline void sortRows(Mat &m)
	{
		deepSort<T>(m, 0, 0, m.rows - 1);
	}

	// ------------------------------------------- SUFFLE ------------------------------------------
	// ----------------------- fast random shuffle of Mat elements with PPL  -----------------------
	/**
	* @brief Randomly shuffles the rows of the input matrix.
	* @details > This function supports PPL.
	* > When using PPL, the result of this function is biased.
	* @param[in,out] m The input/output data, which rows should be shffled.
	* @todo Eliminate the bias, caused by parallel processing.
	*/
	DllExport inline void shuffleRows(Mat &m)
	{
#ifdef ENABLE_PPL
		// int nCores = MAX(1, std::thread::hardware_concurrency());
		int nCores = MAX(1, concurrency::CurrentScheduler::Get()->GetNumberOfVirtualProcessors());
		int step = MAX(2, m.rows / (nCores * 10));
		concurrency::parallel_for(0, m.rows, step, [step, &m](int S) {
			Mat tmp;
			int last = MIN(S + step, m.rows);
			for (int s = last - 1; s > S; s--) {									// s = [last - 1; S + 1]
				dword r = DirectGraphicalModels::random::u<dword>(S, s);			// r = [S; s] = [S; S + 1] -> [S; last - 1]
				if (r != s) Swap(m.row(s), m.row(r), tmp);
			}
		});
#else	
		Mat tmp;
		for (int s = m.rows - 1; s > 0; s--) {			// s = [n-1; 1]
			int r = random::u<int>(0, s);				// r = [0; s] = [0; 1] -> [0; n-1]
			if (r != s)	Swap(m.row(s), m.row(r), tmp);
		}
#endif
	}

} }
