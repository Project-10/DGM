#include "Coordinate.h"
#include "LinearMapper.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CCoordinate::get(const Mat &img, coordinateType type)
{
	Mat res(img.size(), CV_8UC1);
	int width	= img.cols;
	int height	= img.rows;
	float max	= -1.0f;

	for (register int y = 0; y < height; y++) {
		byte *pRes = res.ptr<byte>(y);
		for (register int x = 0; x < width; x++) {
			switch (type) {
				case COORDINATE_ORDINATE:	pRes[x] = linear_mapper<byte>(static_cast<float>(y), 0, static_cast<float>(height - 1)); break;
				case COORDINATE_ABSCISS:	pRes[x] = linear_mapper<byte>(static_cast<float>(x), 0, static_cast<float>(width  - 1)); break;
				case COORDINATE_RADIUS:		
					float dx = x - 0.5f * width;
					float dy = y - 0.5f * height;
					float val = sqrtf(dx*dx + dy*dy);
					if (max < 0) max = val;
					pRes[x] = linear_mapper<byte>(val, 0, max);
			} // type
		} // x
	} // y
	
	return res;
}
} }
