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
		* @brief asd
		* @details asd
		* @tparam Targ asd
		* @tparam Tres asd
		* @param a asd
		* @param b asd
		* @returns asd
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
	}
}