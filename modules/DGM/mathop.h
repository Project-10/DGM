// Mathematical Operations
// Written by Sergey G. Kosov in 2017 for Project X
#pragma once

#include "types.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	// ================================ Random Namespace ==============================
	/**
	* @brief Mathematical operations
	* @details This namespace collects some mathematical operations on matrices
	* @author Sergey G.Kosov, sergey.kosov@project-10.de
	*/
	namespace mathop {
		/**
		* @brief Calculates the Euclidian distance between argument matrices \b a and \b b.
		* @details The Euclidian distance is calculated by the formula : \f$D_E(a, b) = \sqrt{ \sum_{i,j}(a_{ij} - b_{ij})^2 }\f$.
		* @tparam Targ Type of elements in matrices \b a and \b b (\a e.g. \a byte, \a float, \a double, \a etc.)
		* @tparam Tres Type of the resulting distance (\a float or \a double)
		* @param a The first matrix
		* @param b The second matrix
		* @returns The Euclidian distance : \f$D_E(a, b)\f$
		*/
		template <typename Targ = float, typename Tres = float>
		inline Tres Euclidian(const Mat &a, const Mat &b)
		{
			// Assertions
			DGM_ASSERT_MSG(a.size() == b.size(), "Size mismatch");
			DGM_ASSERT_MSG(a.type() == b.type(), "Type mismatch");

			Tres res = 0;
			if (a.cols == 1) {
				for (register int j = 0; j < a.rows; j++)
					res += (static_cast<Tres>(a.at<Targ>(j, 0)) - static_cast<Tres>(b.at<Targ>(j, 0))) * (static_cast<Tres>(a.at<Targ>(j, 0)) - static_cast<Tres>(b.at<Targ>(j, 0)));
			}
			else if (a.rows == 1) {
				for (register int i = 0; i < a.cols; i++) {
					res += (static_cast<Tres>(a.at<Targ>(0, i)) - static_cast<Tres>(b.at<Targ>(0, i))) * (static_cast<Tres>(a.at<Targ>(0, i)) - static_cast<Tres>(b.at<Targ>(0, i)));
				}
			}
			else {
				for (register int j = 0; j < a.rows; j++) {
					const Targ *pa = a.ptr<Targ>(j);
					const Targ *pb = b.ptr<Targ>(j);
					for (register int i = 0; i < a.cols; i++)
						res += (static_cast<Tres>(pa[i]) - static_cast<Tres>(pb[i])) * (static_cast<Tres>(pa[i]) - static_cast<Tres>(pb[i]));
				}
			}
			res = sqrt(res);
			return res;

		}

		template <typename T>
		inline T Euclidian(const std::vector<T> &P, const std::vector<T> &Q)
		{
			T sum = 0;
			for (size_t i = 0; i < P.size(); i++)
				sum += (P[i] - Q[i]) * (P[i] - Q[i]);
			return sqrt(sum);
		}
	}
}