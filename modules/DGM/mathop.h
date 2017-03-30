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
		* @brief Compares two argument matrices \b a and \b b.
		* @tparam T Type of elements in matrices \b a and \b b (\a e.g. \a byte, \a float, \a double, \a etc.)
		* @param a The first matrix
		* @param b The second matrix
		* @retval true if \b a == \b b
		* @retval false otherwise
		*/
		template <typename T>
		inline bool isEqual(const Mat &a, const Mat &b)
		{
			// Assertions
			DGM_ASSERT_MSG(a.size() == b.size(), "Size mismatch");
			DGM_ASSERT_MSG(a.type() == b.type(), "Type mismatch");

			if (a.cols == 1) {
				for (register int j = 0; j < a.rows; j++)
					if (a.at<T>(j, 0) != b.at<T>(j, 0)) return false;
			}
			else if (a.rows == 1) {
				for (register int i = 0; i < a.cols; i++) {
					if (a.at<T>(0, i) != b.at<T>(0, i)) return false;
				}
			}
			else {
				for (register int j = 0; j < a.rows; j++) {
					const T *pa = a.ptr<T>(j);
					const T *pb = b.ptr<T>(j);
					for (register int i = 0; i < a.cols; i++)
						if(pa[i] != pb[i]) return false;
				}
			}
			return true;
		}
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
		inline T Euclidian(const std::vector<T> &a, const std::vector<T> &b)
		{
			T sum = 0;
			for (size_t i = 0; i < a.size(); i++)
				sum += (a[i] - b[i]) * (a[i] - b[i]);
			return sqrt(sum);
		}

		/**
		* @brief Checks whether two regions \b box1 and \b box2 overlap each other
		* @tparam T The type of the elemets, defining the regions \b box1 and \b box2
		* @retval true if regions overlap
		* @retval false otherwise
		*/
		template<typename T>
		bool ifOverlap(pair_mat_t &box1, pair_mat_t &box2)
		{
			for (int x = 0; x < box1.first.cols; x++) {
				if (box1.first.at<T>(0, x) > box2.second.at<T>(0, x))	return false;
				if (box1.second.at<T>(0, x) < box2.first.at<T>(0, x))	return false;
			}
			return true;
		}
	}
}