#include "MarkerHistogram.h"
#include "DGM\TrainNodeNaiveBayes.h"
#include "DGM\IPDF.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace vis 
{
// Constants
const CvSize		CMarkerHistogram::margin		= cvSize(25, 16);
const byte			CMarkerHistogram::bkgIntencity	= 50;
const double		CMarkerHistogram::frgWeight		= 0.75;
const std::string	CMarkerHistogram::wndName		= "Feature Histogram Viewer";

Mat	CMarkerHistogram::drawClassificationMap2D(void) const
{
	char			str[256];
	const byte		nStates		= m_pNodeTrainer->getNumStates();
	const word		nFeatures	= m_pNodeTrainer->getNumFeatures();
	const float		koeff		= 100.0f; // 1e-32f;
	const size_t	n			= m_vPalette.size();

	Mat		res(2 * margin.height + 256, 2 * margin.height + 256, CV_8UC3);	
	res.setTo(bkgIntencity);
	rectangle(res, Point(margin.height, margin.height), Point(256 + margin.height, 256 + margin.height), CV_RGB(0, 0, 0), -1);

	if (nFeatures == 2) {
#ifdef ENABLE_PPL
		concurrency::parallel_for(0, 256, [&](int y) {
#else
		for (int y = 0; y < 256; y++) {
#endif
			Mat fv(2, 1, CV_8UC1);
			fv.at<byte>(1, 0) = static_cast<byte>(y);
			Vec3b *pRes = res.ptr<Vec3b>(margin.height + 255 - y);
			for (int x = 0; x < 256; x++) {
				fv.at<byte>(0, 0) = static_cast<byte>(x);
				Mat pot = m_pNodeTrainer->getNodePotentials(fv);

				for (int s = 0; s < pot.rows; s++) {
					float val = MIN(100, koeff * pot.at<float>(s, 0));
					Scalar color = val * m_vPalette[s % n].first / 100;
					pRes[margin.height + x] += Vec3b((byte)color[0], (byte)color[1], (byte)color[2]);
				}
			} // x
		} // y
#ifdef ENABLE_PPL
		);
#endif
	}
	else DGM_WARNING("The number of features (%d) is not 2", nFeatures);

	// Feature Names
	Mat tmp(margin.height, res.rows, CV_8UC3);
	tmp.setTo(bkgIntencity);
	sprintf(str, "                        feature 1                    255");
	putText(tmp, str, Point(3, tmp.rows - 4), FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(255, 255, 255), 1, CV_AA);
	flip(tmp.t(), tmp, 0);
	tmp.copyTo(res(Rect(0, 0, margin.height, tmp.rows)));
	sprintf(str, "0                       feature 0                    255");
	putText(res, str, Point(3, res.rows - 6), FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(255, 255, 255), 1, CV_AA);

	// Figure box
	rectangle(res, Point(margin.height - 1, margin.height - 1), Point(256 + margin.height, 256 + margin.height), CV_RGB(255, 255, 255));

	return res;
}

void CMarkerHistogram::showHistogram(void)
{
	static Mat histogramImg = drawHistogram(CV_RGB(0, 0, 0));
	namedWindow(wndName.c_str(), WINDOW_AUTOSIZE);
	imshow(wndName.c_str(), histogramImg);
	setMouseCallback(wndName.c_str(), [](int Event, int x, int y, int flags, void *param) {
		if (Event != CV_EVENT_LBUTTONDOWN) return;
		CMarkerHistogram *pUserData = static_cast<CMarkerHistogram *>(param);
		Vec3b color = histogramImg.at<Vec3b>(y, x);	// BGR
		histogramImg.release();
		histogramImg = pUserData->drawHistogram(CV_RGB(color[2], color[1], color[0]));
		imshow(wndName.c_str(), histogramImg);		
	}, this);
}

void CMarkerHistogram::close(void) const
{
	destroyWindow(wndName.c_str());
}

// ======================================== Private ========================================

Mat CMarkerHistogram::drawHistogram(Scalar color) const
{
	const byte		fMaxHeight = 9;					// The maximal number of feature histograms in a column
	const word		nFeatures = m_pNodeTrainer->getNumFeatures();
	const int		activeState = getActiveState(color);

	CvSize			fSize;								// Size of the resulting image in feature histograms
	fSize.width = nFeatures / fMaxHeight;
	if (nFeatures % fMaxHeight != 0) fSize.width++;
	fSize.height = (nFeatures < fMaxHeight) ? nFeatures : fMaxHeight;

	CvSize	resSize;									// Size of the resulting image
	resSize.width = margin.width + fSize.width  * (256 + 2 * margin.width);
	resSize.height = margin.height + fSize.height * (100 + margin.height);

	// Legende
	Mat legende = drawLegend(resSize.height - margin.height, activeState);
	resSize.width += legende.cols;

	Mat	res(resSize, CV_8UC3);							// Resulting Image
	res.setTo(bkgIntencity);

	for (word f = 0; f < nFeatures; f++) {				// freatures
		int dx = f / fMaxHeight;	dx *= (256 + 2 * margin.width);
		int dy = f % fMaxHeight;	dy *= (100 + margin.height);

		Mat featureHistogram = drawFeatureHistogram(f, activeState);
		Rect roi(Point(dx, margin.height + dy), featureHistogram.size());
		featureHistogram.copyTo(res(roi));
		featureHistogram.release();
	} // f


	Rect roi(Point(res.cols - legende.cols, margin.height), legende.size());
	legende.copyTo(res(roi));
	legende.release();

	return res;
}

Mat	CMarkerHistogram::drawHistogram2D(Scalar color) const
{
	const int activeState = getActiveState(color);
	return drawFeatureHistogram2D(0, activeState);
}

int CMarkerHistogram::getActiveState(Scalar color) const
{
	size_t nStates = m_vPalette.size();
	for (byte s = 0; s < nStates; s++) {						// states
		Scalar diff = (Scalar)CV_RGB(bkgIntencity, bkgIntencity, bkgIntencity) + frgWeight * m_vPalette[s].first - color;
		if (norm(diff) < 1.0) return s;
	}
	return -1;
}

Mat CMarkerHistogram::drawFeatureHistogram(word f, int activeState) const
{
	int				x, y;
	char			str[256];
	const byte		nStates	= m_pNodeTrainer->getNumStates();
	const int		koeff	= 1200;						// coefficient for histogram value enlargement
	const size_t	n		= m_vPalette.size();

	Mat		res(margin.height + 100, 2 * margin.width + 256, CV_8UC3);	res.setTo(bkgIntencity);
	Mat		tmp(res.size(), res.type());								tmp.setTo(0);
	
	// axis grid
	for (x = 0; x <= 255; x += 51) {
		y = ((x > 75) && (x < 180)) ? 25 : 0;
		line(tmp, Point(margin.width + x, y), Point(margin.width + x, 100), CV_RGB(50, 50, 50));
	}

	// histogram
	if (typeid(*m_pNodeTrainer) == typeid(CTrainNodeNaiveBayes))
		for (byte s = 0; s < nStates; s++) {				// states
				IPDF *pPDF = dynamic_cast<const CTrainNodeNaiveBayes *>(m_pNodeTrainer)->getPDF(s, f);
				DGM_ASSERT(pPDF);
				for (x = 0; x < 256; x++) {
					int len	=  static_cast<int>(koeff * pPDF->getDensity(x));
					y = 100 - MIN(99, len);
				
					if ((activeState == -1) || (activeState == s % n))
						line(tmp, Point(margin.width + x, 100), Point(margin.width + x, y), m_vPalette.at(s % n).first);
				} // x
			addWeighted(res, 1.0, tmp, frgWeight, 0.0, res);
			tmp.setTo(0);
		} // s
	tmp.release();

	// Feature Names
	if (m_vFeatureNames.empty()) sprintf(str, "feature %d", f);
	else						 sprintf(str, "%s", m_vFeatureNames[f].c_str());
	CvSize textSize = getTextSize(str, CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, NULL);
	putText(res, str, Point(margin.width + (MAX(256 - textSize.width, 108)) / 2, 16), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(125, 125, 125), 1, CV_AA);

	// Figure box
	rectangle(res, Point(margin.width - 1, 0), Point(margin.width + 256, 100), CV_RGB(255,255,255));
	for (x = 0; x <= 255; x += 51) {
		sprintf(str, "%d", x);
		putText(res, str, Point(margin.width + x - 5, 109), FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(255, 255, 255), 1, CV_AA);
	}
	
	return res;
}

Mat CMarkerHistogram::drawFeatureHistogram2D(word f, int activeState) const
{
	char			str[256];
	const byte		nStates = m_pNodeTrainer->getNumStates();
	const word		nFeatures = m_pNodeTrainer->getNumFeatures();
	const int		koeff = 1000000;					// coefficient for histogram value enlargement
	const size_t	n = m_vPalette.size();

	Mat		res(2 * margin.height + 256, 2 * margin.height + 256, CV_8UC3);	res.setTo(bkgIntencity);
//	rectangle(res, Point(margin.height, margin.height), Point(256 + margin.height, 256 + margin.height), CV_RGB(0, 0, 0), -1);
	Mat		tmp(res.size(), res.type());									tmp.setTo(0);

	// histogram
	if ((typeid(*m_pNodeTrainer) == typeid(CTrainNodeNaiveBayes)) && (nFeatures == 2)) {
		for (byte s = 0; s < nStates; s++) {				// states
			IPDF *pPDF2D = dynamic_cast<const CTrainNodeNaiveBayes *>(m_pNodeTrainer)->getPDF2D(s);
			DGM_ASSERT(pPDF2D);
			for (int y = 0; y < 256; y++) {
				Vec3b *pTmp = tmp.ptr<Vec3b>(margin.height + 256 - y);
				for (int x = 0; x < 256; x++) {
					double val = MIN(255, koeff * pPDF2D->getDensity(Scalar(x, y)));
					Scalar color = val * m_vPalette[s % n].first / 255;
					
					if ((activeState == -1) || (activeState == s % n))
						pTmp[margin.height + x] = Vec3b((byte)color[0], (byte)color[1], (byte)color[2]);
				}
			}
			addWeighted(res, 1.0, tmp, frgWeight, 0.0, res);
			tmp.setTo(0);
		} // s
		tmp.release();
	}
	else DGM_WARNING("The node trainer (%s) is not Bayes or the number of features (%d) is not 2", typeid(*m_pNodeTrainer).name(), nFeatures);

	// Feature Names
	tmp = Mat(margin.height, res.rows, CV_8UC3);	
	tmp.setTo(bkgIntencity);
	sprintf(str, "                        feature 1                    255");
	putText(tmp, str, Point(3, tmp.rows - 4), FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(255, 255, 255), 1, CV_AA);
	flip(tmp.t(), tmp, 0);
	tmp.copyTo(res(Rect(0, 0, margin.height, tmp.rows)));
	sprintf(str, "0                       feature 0                    255");
	putText(res, str, Point(3, res.rows - 6), FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(255, 255, 255), 1, CV_AA);

	// Figure box
	rectangle(res, Point(margin.height - 1, margin.height - 1), Point(256 + margin.height, 256 + margin.height), CV_RGB(255, 255, 255));

	return res;
}

Mat CMarkerHistogram::drawLegend(int maxHeight, int activeState) const
{
	char			str[256];
	const byte		sMaxHeight	= maxHeight / (2 * margin.height);					// The maximal number of states in a column
	const byte		nStates		= m_pNodeTrainer->getNumStates();
	const size_t	n			= m_vPalette.size();
	
	CvSize			sSize;								// Size of the resulting image in states
	sSize.width = (nStates + 1) / sMaxHeight;
	if ((nStates + 1) % sMaxHeight != 0) sSize.width++;
	sSize.height = ((nStates + 1) < sMaxHeight) ? (nStates + 1) : sMaxHeight;

	CvSize			resSize;							// Size of the resulting image
	resSize.width  = sSize.width  * 120;
	resSize.height = 2 * margin.height * sSize.height;

	Mat res(resSize,  CV_8UC3);	res.setTo(bkgIntencity);
	Mat tmp(res.size(), res.type());							tmp.setTo(0);

	for (byte s = 0; s < nStates; s++) {				// states
		int dx = s / sMaxHeight;	dx *= 120;
		int dy = s % sMaxHeight;	dy *= 2 * margin.height;

		// Color box
		rectangle(tmp, Point(dx, dy), Point(dx + margin.width, dy + margin.height), m_vPalette.at(s % n).first, -1);
		
		// Active triangle
		if (activeState == s)  {
			const Point triangle[3] = {Point(dx, dy), Point(dx + margin.height / 2, dy + margin.height / 2), Point(dx, dy + margin.height)}; 
			fillConvexPoly(tmp, triangle, 3, CV_RGB(0, 0, 0));
		}
		
		// Class name
		if (m_vPalette.at(s % n).second.empty()) sprintf(str, "c%d", s);
		else  sprintf(str, "%s", m_vPalette.at(s % n).second.c_str());
		putText(tmp, str, Point(dx + margin.width + 5, dy + margin.height - 3), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 1, CV_AA);		
	} // s
	
	// White "all" color box
	int dx = nStates / sMaxHeight;	dx *= 120;
	int dy = nStates % sMaxHeight;	dy *= 2 * margin.height;
	rectangle(tmp, Point(dx, dy), Point(dx + margin.width, dy + margin.height), CV_RGB(255, 255, 255), -1);
	if (activeState == -1)  {
		const Point triangle[3] = {Point(dx, dy), Point(dx + margin.height / 2, dy + margin.height / 2), Point(dx, dy + margin.height)}; 
		fillConvexPoly(tmp, triangle, 3, CV_RGB(0, 0, 0));
	}	
	putText(tmp, "all", Point(dx + margin.width + 5, dy + margin.height - 3), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 255, 255), 1, CV_AA);

	addWeighted(res, 1.0, tmp, frgWeight, 0.0, res);
	tmp.release();

	return res;
}

} }