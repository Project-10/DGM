#include "Variance.h"
#include "LinearMapper.h"

namespace DirectGraphicalModels { namespace fex
{
Mat	CVariance::get(const Mat &img, SqNeighbourhood nbhd)
{
	int				width	= img.cols;
	int				height	= img.rows;

	// Converting to one channel image
	Mat	I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I);

	Mat res(img.size(), CV_8UC1);									
	Mat integralImg;												
	integral(I, integralImg);
	
	for (register int y = 0; y < height; y++) {
		int		 y0		= MAX(0, y - nbhd.upperGap);
		int	 	 y1		= MIN(y + nbhd.lowerGap, height -1);
		byte	*pRes	= res.ptr<byte>(y);
		int *pI0	= integralImg.ptr<int>(y0);
		int *pI1	= integralImg.ptr<int>(y1 + 1);
		for (register int x = 0; x < width; x++) {
			int		x0	= MAX(0, x - nbhd.leftGap);
			int		x1	= MIN(x + nbhd.rightGap, width - 1);
			int		S	= (x1 - x0 + 1) * (y1 - y0 + 1);
			float	med = static_cast<float>(pI1[x1 + 1] - pI1[x0] - pI0[x1 + 1] + pI0[x0]) / S;
			float	Sum = 0;
			for (register int j = y0; j <= y1; j++) {
				byte *pImg = I.ptr<byte>(j);
				for (register int i = x0; i <= x1; i++) {
					float dif = static_cast<float>(pImg[i]) - med;
					Sum += dif*dif;
				} // i
			} // j
			float val = sqrtf(Sum / S);
			pRes[x] = linear_mapper<byte>(val, 0, 100);
		} // x
	} // y

	return res;	
}
} }