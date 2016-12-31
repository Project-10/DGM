#include "HOG.h"
#include "Gradient.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace fex
{
Mat CHOG::get(const Mat &img, int nBins, SqNeighbourhood nbhd)
{
	DGM_ASSERT_MSG(nBins < CV_CN_MAX, "Number of bins (%d) exceeds the maximum allowed number (%d)", nBins, CV_CN_MAX);
	
	register int	i;						// bins index
	register int	x, y;
	int				width	= img.cols;
	int				height	= img.rows;

	// Converting to one channel image
	Mat	I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I);
	
	// Derivatives
	Mat Ix = CGradient::getDerivativeX(I);
	Mat Iy = CGradient::getDerivativeY(I);

	// Initializing bins and integrals
	vec_mat_t vTemp(nBins);
	vec_mat_t vBins(nBins);
	vec_mat_t vInts(nBins);

	for (i = 0; i < nBins; i++) {
		vTemp[i].create(img.size(), CV_8UC1);
		vBins[i].create(img.size(), CV_32FC1);	
		vBins[i].setTo(0);
	}

	std::vector<float *>	pBins(nBins);
	std::vector<double *>	pInts0(nBins);
	std::vector<double *>	pInts1(nBins);
	std::vector<byte *>		pTemp(nBins);
	
	// Caclculating the bins
	for (y = 0; y < height; y++) {
		float *pIx = Ix.ptr<float>(y);
		float *pIy = Iy.ptr<float>(y);
		for (i = 0; i < nBins; i++) pBins[i] = vBins[i].ptr<float>(y);
		for (x = 0; x < width; x++) {
			float ix = pIx[x];
			float iy = pIy[x];
			
			// gradient Magnitude
			float gMgn = sqrtf(ix*ix + iy*iy);

			// gradient Orientation
			if (fabs(ix) < FLT_EPSILON) ix = SIGN(ix) * FLT_EPSILON;
			float tg = iy / ix;
			float gOrt = (0.5f + atanf(tg) / (float)Pi) * 180.0f;			// [0°; 180°]

			// filling in the bins
			float gOrtStep = 180.0f / nBins;
			for (i = 0; i < nBins; i++)
				if (gOrt <= (i + 1) * gOrtStep) {
					pBins[i][x] = gMgn;
					break;
				}
		}
	}

	// Calculating the integrals
	for (i = 0; i < nBins; i++) integral(vBins[i], vInts[i]);
	
	for (y = 0; y < height; y++) {	
		int y0 = MAX(0, y - nbhd.upperGap);		
		int y1 = MIN(y + nbhd.lowerGap, height - 1);
		for (i = 0; i < nBins; i++) pInts0[i] = vInts[i].ptr<double>(y0);
		for (i = 0; i < nBins; i++) pInts1[i] = vInts[i].ptr<double>(y1 + 1);
		for (i = 0; i < nBins; i++) pTemp[i]  = vTemp[i].ptr<byte>(y);
		for (x = 0; x < width; x++) {
			int x0 = MAX(0, x - nbhd.leftGap);
			int x1 = MIN(x + nbhd.rightGap, width - 1);

			Mat HOGcell(cvSize(nBins, 1), CV_64FC1);
			double *pHOGcell = HOGcell.ptr<double>(0);
			for (i = 0; i < nBins; i++) pHOGcell[i] = pInts1[i][x1 + 1] - pInts1[i][x0] - pInts0[i][x1 + 1] + pInts0[i][x0];
			normalize(HOGcell, HOGcell, 255, 0, CV_MINMAX);
			for (i = 0; i < nBins; i++) pTemp[i][x] = static_cast<byte>(pHOGcell[i]);
			HOGcell.release();
		} // x
	} // y

	Mat res;
	merge(vTemp, res);

	return res;	
}
} }