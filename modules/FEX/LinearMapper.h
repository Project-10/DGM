// Linear Mapper set of functions 
// Written by Sergey G. Kosov in 2015 - 2016 for Project X
#pragma once

#include "types.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
	/**
	* @brief Linear 1D mapping.
	* @details This function perform linear mapping of the value \b val from one interval to another: \f$val\in[min; max]\rightarrow res\in[T.min; T.max]\f$, such that:
	* \f{eqnarray*}{ min&\rightarrow&T.min \\ max&\rightarrow&T.max \f}
	* @tparam T The type of the resulting value. Usually \a byte or \a word. It also defines the resulting interval, 
	* \a e.g for \a byte it is [0; 255] and for \a word it is [0; 65535].
	* @param val The value to map.
	* @param min The lower boundary of the \b val.
	* @param max The higher bounday of the \b val.
	* @returns The mapped value \b res.
	*/
	template <typename T>
	inline T linear_mapper(float val, float min, float max)
	{
		DGM_ASSERT(max > min);
		float tMin = static_cast<float>(std::numeric_limits<T>::min());
		float tMax = static_cast<float>(std::numeric_limits<T>::max());
		float a = (tMax - tMin) / (max - min);
		float b = tMin - a * min;
		float x = a * val + b;
		return static_cast<T>(MAX(tMin, MIN(tMax, std::round(x))));
	}
	
	/**
	* @brief Two-linear 1D mapping.
	* @details This function perform linear mapping of the value \b val from one interval to another: \f$val\in[min; max]\rightarrow res\in[T.min; T.max]\f$, such that:
	* \f{eqnarray*}{ min&\rightarrow&T.min \\ mid&\rightarrow&midPoint \\ max&\rightarrow&T.max \f}
	* For more detail please refer to the \b Figure \b 1.
	* @image html two_linear_mapping.gif "Fig. 1"
	* @param val The value to map.
	* @param min The lower boundary of the \b val.
	* @param max The higher bounday of the \b val.
	* @param mid The x-coordinate of the intersection point, \f$mid\in(min; max)\f$ (Ref. \b Figure \b 1).
	* @param midPoint The y-coordinate of the intersection point, \f$midPoint\in[T.min; T.max]\f$ (Ref. \b Figure \b 1).
	* @returns The mapped value \b res.
	*/
	template<typename T>
	inline T two_linear_mapper(float val, float min, float max, float mid, T midPoint)
	{
		if ((mid == min) || (mid == max)) return linear_mapper<T>(val, min, max);
		DGM_ASSERT(max > mid);
		DGM_ASSERT(mid > min);
		float tMin = static_cast<float>(std::numeric_limits<T>::min());
		float tMid = static_cast<float>(midPoint);
		float tMax = static_cast<float>(std::numeric_limits<T>::max());

		float a, b;
		if (val < mid) {
			a = (tMid - tMin) / (mid - min);
			b = tMin - a * min;
		}
		else {
			a = (tMax - tMid) / (max - mid);
			b = tMid - a * mid;
		}

		float x = a * val + b;
		return static_cast<T>(MAX(tMin, MIN(tMax, std::round(x))));
	}

} }