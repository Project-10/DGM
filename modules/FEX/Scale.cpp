#include "Scale.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CScale::get(const Mat &img, SqNeighbourhood nbhd)
{
	int	width		= img.cols;
	int	height		= img.rows;
	int	nChannels	= img.channels();

	Mat res(img.size(), img.type());
	vec_mat_t vChannels;
	split(img, vChannels);

	Mat integralImg;												
	for (byte c = 0; c < nChannels; c++) {		
		integral(vChannels[c], integralImg);

		for (int y = 0; y < height; y++) {
			int		 y0 = MAX(0, y - nbhd.upperGap);
			int	 	 y1 = MIN(y + nbhd.lowerGap, height - 1);
			byte	*pRes = res.ptr<byte>(y);
			int *pI0 = integralImg.ptr<int>(y0);
			int *pI1 = integralImg.ptr<int>(y1 + 1);
			for (int x = 0; x < width; x++) {
				int		x0 = MAX(0, x - nbhd.leftGap);
				int		x1 = MIN(x + nbhd.rightGap, width - 1);
				int		S = (x1 - x0 + 1) * (y1 - y0 + 1);
				float	med = static_cast<float>(pI1[x1 + 1] - pI1[x0] - pI0[x1 + 1] + pI0[x0]) / S;
				pRes[nChannels * x + c] = static_cast<byte>(med + 0.5f);
			} // x
		} // y
	} // c

	return res;	
}
} }