#include "BaseFeatureExtractor.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
byte CBaseFeatureExtractor::linear_mapper(float val, float min, float max)
{
	DGM_ASSERT(max > min);
	return static_cast<byte> (MIN(255, MAX(0, 255 * (val - min) / (max - min) + 0.5f)));
}

byte CBaseFeatureExtractor::two_linear_mapper(float val, float min, float max, float mid, byte midPoint)
{
 	if ((mid == min) || (mid == max)) return linear_mapper(val, min, max);
	DGM_ASSERT(max > mid);
	DGM_ASSERT(mid > min);
	
	float x		= 255 * (val - min) / (max - min);
	float x0	= 255 * (mid - min) / (max - min);
	float y0	= static_cast<float>(midPoint);

	float a, b;
	if (x < x0) {
		a = y0 / x0;
		b = 0;
	} else {
		a = (255 - y0) / (255 - x0);
		b = 255 * (y0  - x0) / (255 - x0);
	}
	float X = a * x + b;
	
	byte res = static_cast<byte> (MIN(255, MAX(0, X + 0.5f)));
	return res;
}
} }