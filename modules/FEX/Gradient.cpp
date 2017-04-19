#include "Gradient.h"
#include "LinearMapper.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CGradient::get(const Mat &img, float mid)
{
	DGM_ASSERT(mid <= GRADIENT_MAX_VALUE);
	DGM_ASSERT(mid > 0);
	 
	// Converting to one channel image
	Mat I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I);
	
	// Derivatives
	Mat Ix = getDerivativeX(I);
	Mat Iy = getDerivativeY(I);

	// Magnitude
	Mat res(img.size(), CV_8UC1);		// gradient 	
	for(register int y = 0; y < img.rows; y++) {	
		float *pIx	= Ix.ptr<float>(y);
		float *pIy	= Iy.ptr<float>(y);
		for(register int x = 0; x < img.cols; x++) {
			float val = sqrtf(pIx[x]*pIx[x] + pIy[x]*pIy[x]);
			res.at<byte>(y,x) = two_linear_mapper<byte>(val, 0, GRADIENT_MAX_VALUE, mid, 255);
		}
	} // y	

	return res;
}

Mat CGradient::getDerivativeX(const Mat &img)
{
	DGM_ASSERT(img.channels() == 1);
	
	Mat res(img.size(), CV_32FC1); res.setTo(0);
	for(register int y = 0; y < res.rows; y++) {
		const byte	*pImg	= img.ptr<byte>(y);
		float		*pRes	= res.ptr<float>(y);
		for(register int x = 1; x < res.cols - 1; x++)
			pRes[x] = 0.5f * (static_cast<float>(pImg[x + 1]) - static_cast<float>(pImg[x - 1]));
	} // y
	return res;
}

Mat CGradient::getDerivativeY(const Mat &img)
{
	DGM_ASSERT(img.channels() == 1);
	
	Mat res(img.size(), CV_32FC1); res.setTo(0);
	for(register int y = 1; y < res.rows - 1; y++) {
		const byte	*pImgF	= img.ptr<byte>(y + 1);
		const byte	*pImgB	= img.ptr<byte>(y - 1);
		float		*pRes	= res.ptr<float>(y);
		for(register int x = 0; x < res.cols; x++)
			pRes[x] = 0.5f * (static_cast<float>(pImgF[x]) - static_cast<float>(pImgB[x]));
	} // y
	return res;
}

} }