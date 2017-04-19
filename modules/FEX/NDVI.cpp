#include "NDVI.h"
#include "LinearMapper.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CNDVI::get(const Mat &img, byte midPoint)
{
	DGM_ASSERT_MSG(img.channels() == 3, "Input image has %d channel(s), but must have 3.", img.channels());
	Mat res(img.size(), CV_8UC1);
	vec_mat_t vChannels;
	split(img, vChannels);	

	for (register int y = 0; y < res.rows; y++) {
		byte *pRes	= res.ptr<byte>(y);
		byte *pR	= vChannels.at(2).ptr<byte>(y);
		byte *pG	= vChannels.at(1).ptr<byte>(y);
		byte *pB	= vChannels.at(0).ptr<byte>(y);
		for (register int x = 0; x < res.cols; x++) {
			float nir	= static_cast<float>(pR[x]);
			float vis	= 0.5f * (static_cast<float>(pG[x]) + static_cast<float>(pB[x]));
			float ndvi = (nir + vis > 0) ? (nir - vis) / (nir + vis) : 0;

			pRes[x] = two_linear_mapper<byte>(ndvi, -1.0f, 1.0f, 0.0f, midPoint);
		}
	} // y

	return res;
}
} }