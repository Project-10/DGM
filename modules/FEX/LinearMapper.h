// Linear Mapper set of functions 
// Written by Sergey G. Kosov in 2015 - 2016 for Project X
#pragma once

#include "types.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
	/**
	* @brief Linear 1D mapping.
	* @details This function perform linear mapping of the value \b val from one interval to another: \f$val\in[min; max]\rightarrow res\in[0; 255]\f$, such that:
	* \f{eqnarray*}{ min&\rightarrow&0 \\ max&\rightarrow&255 \f}
	* @param val The value to map.
	* @param min The lower boundary of the \b val.
	* @param max The higher bounday of the \b val.
	* @returns The mapped value \b res.
	*/
	inline byte linear_mapper(float val, float min, float max)
	{
		DGM_ASSERT(max > min);
		return static_cast<byte> (MIN(255, MAX(0, 255 * (val - min) / (max - min) + 0.5f)));
	}
	
	/**
	* @brief Two-linear 1D mapping.
	* @details This function perform linear mapping of the value \b val from one interval to another: \f$val\in[min; max]\rightarrow res\in[0; 255]\f$, such that:
	* \f{eqnarray*}{ min&\rightarrow&0 \\ mid&\rightarrow&midPoint \\ max&\rightarrow&255 \f}
	* For more detail please refer to the \b Figure \b 1.
	* @image html two_linear_mapping.gif "Fig. 1"
	* @param val The value to map.
	* @param min The lower boundary of the \b val.
	* @param max The higher bounday of the \b val.
	* @param mid The x-coordinate of the intersection point, \f$mid\in(min; max)\f$ (Ref. \b Figure \b 1).
	* @param midPoint The y-coordinate of the intersection point, \f$midPoint\in[0; 255]\f$ (Ref. \b Figure \b 1).
	* @returns The mapped value \b res.
	*/
	inline byte two_linear_mapper(float val, float min, float max, float mid, byte midPoint)
	{
		if ((mid == min) || (mid == max)) return linear_mapper(val, min, max);
		DGM_ASSERT(max > mid);
		DGM_ASSERT(mid > min);

		float x = 255 * (val - min) / (max - min);
		float x0 = 255 * (mid - min) / (max - min);
		float y0 = static_cast<float>(midPoint);

		float a, b;
		if (x < x0) {
			a = y0 / x0;
			b = 0;
		}
		else {
			a = (255 - y0) / (255 - x0);
			b = 255 * (y0 - x0) / (255 - x0);
		}
		float X = a * x + b;

		byte res = static_cast<byte> (MIN(255, MAX(0, X + 0.5f)));
		return res;
	}

} }