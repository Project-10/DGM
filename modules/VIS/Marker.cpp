#include "Marker.h"
#include "ColorSpaces.h"
#include "macroses.h"


namespace DirectGraphicalModels { namespace vis
{
// Constants
const byte	CMarker::bkgIntencity	= 222;
const byte	CMarker::frgIntensity	= 255;
const int	CMarker::ds				= 70; // px

// Constuctor
CMarker::CMarker(default_pallete palette)
{
	switch(palette) {
		case DEF_PALETTE_3:			for (int h = 0; h < 360; h += 120)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2rgb(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_3_INV:		for (int h = 0; h < 360; h += 120)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2bgr(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_6:			for (int h = 0; h < 360; h += 60)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2rgb(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_6_INV:		for (int h = 0; h < 360; h += 60)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2bgr(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_12:		for (int h = 0; h < 360; h += 30)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2rgb(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_12_INV:	for (int h = 0; h < 360; h += 30)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2bgr(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_24:		for (int h = 0; h < 360; h += 15)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2rgb(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_24_INV:	for (int h = 0; h < 360; h += 15)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2bgr(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_24_M:		for (int i = 0; i < 24; i++)		m_vPalette.push_back(std::make_pair(colors24[i], ""));								  break;
		case DEF_PALETTE_36:		for (int h = 0; h < 360; h += 10)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2rgb(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_36_INV:	for (int h = 0; h < 360; h += 10)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2bgr(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_72:		for (int h = 0; h < 360; h += 5)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2rgb(DGM_HSV(h, 255, 255)), "")); break;
		case DEF_PALETTE_72_INV:	for (int h = 0; h < 360; h += 5)	m_vPalette.push_back(std::make_pair(colorspaces::hsv2bgr(DGM_HSV(h, 255, 255)), "")); break;
	}	
}

// Constuctor
CMarker::CMarker(const vec_nColor_t &vPalette) : m_vPalette(vPalette)
{}

// Destructor:
CMarker::~CMarker(void)
{
	m_vPalette.clear();
}

void CMarker::markClasses(Mat &base, const Mat &classes, byte flag) const
{
	size_t	n		= m_vPalette.size();
	bool	mapping = true;
	if (base.empty()) {
		base = Mat(classes.size(), CV_8UC3);
		base.setTo(0);
		mapping = false;
	}
	
	// Desaturating
	if (mapping && !((flag & MARK_OVER) == MARK_OVER)) { // desaturate
		cvtColor(base, base, CV_BGR2GRAY);
		cvtColor(base, base, CV_GRAY2RGB);
	}

	// Assertions
	DGM_ASSERT_MSG(base.channels() == 3, "Base image has %d channel(s), but must have 3.", base.channels());
	DGM_ASSERT_MSG(classes.channels() == 1, "Class Map has %d channel(s), but must have 1.", classes.channels()); 

	for (register int y = 0; y < base.rows; y++) {
		byte	   * pBase  = base.ptr<byte>(y);
		const byte * pClass = classes.ptr<byte>(y);
		for (register int x = 0; x < base.cols; x++) {
			if (((flag & MARK_NO_ZERO) == MARK_NO_ZERO) && (pClass[x] == 11)) continue;
			//if (((flag & MARK_GRID) == MARK_GRID) && ((x+y) % 2 == 0)) continue;
			for (int c = 0; c < 3; c++)
				if (mapping && !((flag & MARK_OVER) == MARK_OVER))	pBase[3 * x + c] = static_cast<byte>((pBase[3 * x + c] + m_vPalette.at(pClass[x] % n).first.val[c]) / 2);
				else			pBase[3 * x + c] = static_cast<byte>(m_vPalette.at(pClass[x] % n).first.val[c]);
		} // x
	} // y
}

Mat CMarker::drawPotentials(const Mat &potential, byte flag) const
{
	DGM_ASSERT(!potential.empty());

	if (potential.dims == 2) {
		if (potential.cols == 1)	return drawVector(potential, flag);	// Node potential 
		else 						return drawMatrix(potential, flag);	// Edge potential
	} else							return drawVoxel(potential, flag);	// Triplet potential 		

}

Mat	CMarker::drawConfusionMatrix(const Mat &confusionMat, byte flag) const
{
	const byte		nStates		= confusionMat.rows;
	const CvScalar	frgColor	= CV_RGB(frgIntensity, frgIntensity, frgIntensity); 

	bool perClass	= (flag & MARK_PERCLASS) == MARK_PERCLASS;

	Mat cMat;
	if (perClass) {
		Mat tmp;
		confusionMat.copyTo(tmp);
		for (byte y = 0; y < nStates; y++) {
			float *pTmp = tmp.ptr<float>(y);
			float sum = 0;
			for (byte x = 0; x < nStates; x++) sum += pTmp[x];
			for (byte x = 0; x < nStates; x++) pTmp[x] = (sum > 0.0f) ? 100.0f * pTmp[x] / sum : 0.0f;
		} // y
		cMat = drawMatrix(tmp, flag | TP_PERCENT);
		tmp.release();
	} else cMat = drawMatrix(confusionMat, flag | TP_PERCENT);
	
	CvSize resSize = cMat.size();
	if (!perClass) {
		resSize.width  += ds;
		resSize.height += ds;
	}
	Mat tmp(resSize, CV_8UC3);	tmp.setTo(bkgIntencity);
	Rect roi = Rect(Point(0, 0), cMat.size());
	cMat.copyTo(tmp(roi));
	cMat.release();	

	if (!perClass) {
		drawSquare(tmp, nStates + 1, 0, frgColor, "Recall", 0.45, TP_BOTTOM);
		for (byte y = 0; y < nStates; y++) {
			const float *pCM = confusionMat.ptr<float>(y);
			float sum = 0;
			for (byte x = 0; x < nStates; x++) sum += pCM[x];
			float val = (sum > 0.0f) ? 100.0f * pCM[y] / sum : 0.0f;
			drawSquare(tmp, nStates + 1, y + 1, frgColor, val, 0.35, TP_CENTER | TP_PERCENT);
		} // y
		
		drawSquare(tmp, 0, nStates + 1, frgColor, "Precision", 0.45, TP_RIGHT);
		for (byte x = 0; x < nStates; x++) {
			float sum = 0;
			for (byte y = 0; y < nStates; y++) sum += confusionMat.at<float>(y, x);
			float val = (sum > 0.0f) ? 100.0f * confusionMat.at<float>(x, x) / sum : 0.0f;
			drawSquare(tmp, x + 1, nStates + 1, frgColor, val, 0.35, TP_CENTER | TP_PERCENT);
		} // x
		
		// Overall Accuracy
		float sum = 0;
		for (byte s = 0; s < nStates; s++) sum += confusionMat.at<float>(s, s);
		drawSquare(tmp, nStates + 1, nStates + 1, frgColor, sum, 0.4, TP_CENTER | TP_PERCENT);
	}

	// tmp -> res
	resSize.width  += 25;
	resSize.height += 25;	
	Mat res(resSize, CV_8UC3); res.setTo(bkgIntencity);
	//rectangle(res, Point(0, 0), Point(res.cols, res.rows), CV_RGB(255,0,0), -1);
	roi = Rect(Point(25, 25), tmp.size());
	tmp.copyTo(res(roi));
	tmp.release();


	// Horizontal bar
	drawRectangle(res, Point(1, 1), Point(25 + ds - 1, 25 - 1), frgColor);
	drawRectangle(res, Point(25 + 1 + ds, 1), Point(25 + (nStates + 1) * ds - 1, 25 - 1), frgColor, "Predicted state", 0.45, TP_BOTTOM);
	drawRectangle(res, Point(25 + 1 + (nStates + 1) * ds, 1), Point(25 + (nStates + 2) * ds - 1, 25 - 1), frgColor);

	// Vertical bar
	drawRectangle(res, Point(1, 1), Point(25 - 1, 25 + ds - 1), frgColor);
	tmp = Mat(25 - 1, nStates * ds - 1, CV_8UC3); tmp.setTo(bkgIntencity);
	drawRectangle(tmp, Point(0, 0), tmp.size(), frgColor, "Actual state", 0.45, TP_BOTTOM);
	flip(tmp.t(), tmp, 0);
	roi = Rect(Point(1, 25 + 1 + ds), tmp.size());
	tmp.copyTo(res(roi));
	tmp.release();
	drawRectangle(res, Point(1, 25 + 1 + (nStates + 1) * ds), Point(25 - 1, 25 + (nStates + 2) * ds - 1), frgColor);

	
	return res;
}

// ======================================== Private ========================================

Mat CMarker::drawVector(const Mat &potential, byte flag) const
{
	const byte		nStates		= potential.rows;
	const size_t	n			= m_vPalette.size();

	bool  bw = (flag & MARK_BW) == MARK_BW;

	Size			textSize;
	char			str[256];
	double			max;
	minMaxLoc(potential, NULL, &max);

	Mat res(ds + 1, ds * nStates + 1, CV_8UC3);	res.setTo(bkgIntencity);

	for (byte s = 0; s < nStates; s++) {
		float	potVal	= potential.at<float>(s, 0);

		double	sat = (isnan(potVal)) ? 1.0 : potVal / max;
		Scalar	color_Tgt = (isnan(potVal) || bw) ? CV_RGB(0, 0, 0) : m_vPalette.at(s % n).first;
		Scalar	color_Cur;
		double	intensity = 0;
		for (int i = 0; i < 3; i++) {
			color_Cur.val[i] = sat * color_Tgt.val[i] + (1.0 - sat) * frgIntensity;
			intensity += color_Cur.val[i] / 3;
		}
		drawSquare(res, s, 0, color_Cur, potVal, 0.35, (flag & TP_PERCENT) | TP_BOTTOM); 

		// Text
		color_Cur = (intensity < 128) ? CV_RGB(255, 255, 255) : CV_RGB(0, 0, 0);			

		if (m_vPalette.at(s % n).second.empty()) sprintf(str, "c%d", s);
		else sprintf(str, "%s", m_vPalette.at(s % n).second.c_str()); 
		textSize = getTextSize(str, CV_FONT_HERSHEY_SIMPLEX, 0.45, 1, NULL);
		putText(res, str, Point(ds * s + (MAX(ds - textSize.width, 6)) / 2, 15), FONT_HERSHEY_SIMPLEX, 0.45, color_Cur, 1, CV_AA);		
	} // s
	return res;
}

Mat CMarker::drawMatrix(const Mat &potential, byte flag) const
{
	const byte		nStates		= potential.rows;
	const size_t	n			= m_vPalette.size();
	const CvScalar	frgColor	= CV_RGB(frgIntensity, frgIntensity, frgIntensity); 

	bool  bw = (flag & MARK_BW) == MARK_BW;

	char			str[256];
	double			max;
	minMaxLoc(potential, NULL, &max);

	Mat res(ds * (nStates + 1) + 1, ds * (nStates + 1) + 1, CV_8UC3);	res.setTo(bkgIntencity);
		
	drawSquare(res, 0, 0, frgColor, "");

	for (byte x = 1; x < nStates + 1; x++) {
		if (m_vPalette.at((x - 1) % n).second.empty()) sprintf(str, "c%d", x - 1);
		else sprintf(str, "%s",m_vPalette.at((x - 1) % n).second.c_str()); 
		drawSquare(res, x, 0, frgColor, str, 0.45, TP_BOTTOM);
	} // x
		
	for (byte y = 1; y < nStates + 1; y++) {
		if (m_vPalette.at((y - 1) % n).second.empty()) sprintf(str, "c%d", y - 1);
		else sprintf(str, "%s", m_vPalette.at((y - 1) % n).second.c_str()); 
		drawSquare(res, 0, y, frgColor, str, 0.45, TP_RIGHT);

		const float *pPot = potential.ptr<float>(y - 1);
		for (byte x = 1; x < nStates + 1; x++) {
			float potVal = pPot[x - 1];
			
			double	sat = (isnan(potVal)) ? 1.0 : potVal / max;
			Scalar	color_Tgt = (isnan(potVal) || bw) ? CV_RGB(0, 0, 0) : m_vPalette.at(MAX(0/*x - 1*/, y - 1) % n).first;
			Scalar	color_Cur;
			double	intensity = 0;
			for (int i = 0; i < 3; i++) {
				color_Cur.val[i] = sat * color_Tgt.val[i] + (1.0 - sat) * frgIntensity;
				intensity += color_Cur.val[i] / 3;
			}
			drawSquare(res, x, y, color_Cur, potVal, 0.35, (flag & TP_PERCENT) | TP_CENTER);
		} // x
	} // y
	rectangle(res, Point(ds, ds), Point(ds * (nStates + 1), ds * (nStates + 1)), CV_RGB(0, 0, 0));

	return res;
}

Mat CMarker::drawVoxel(const Mat &potential, byte flag) const
{
	DGM_WARNING("This function is not implemented yet!");
	return Mat();
}

template<typename T> 
void CMarker::drawSquare(Mat &img, byte x, byte y, const Scalar &color, T val, double fontScale, byte textProp) const
{
	Point pt1(1 + x * ds, 1 + y * ds);
	Point pt2((x + 1) * ds - 1, (y + 1) * ds - 1);
	drawRectangle(img, pt1, pt2, color, val, fontScale, textProp);
}

void CMarker::drawRectangle(Mat &img, Point pt1, Point pt2, const Scalar &color, float val, double fontScale, byte textProp) const 
{
	char str[256];
	
	bool procent = (textProp & TP_PERCENT) == TP_PERCENT;

	if (procent) {
		if (isnan(val))								sprintf(str, "N A N");	
		else if (val == 0)							sprintf(str, "O");	
		else if (val < 0.01f)						sprintf(str, "0.00 %%");
		else										sprintf(str, "%3.2f %%", val);	
	} else {
		if (isnan(val))								sprintf(str, "N A N");	
		else if (val == 0)							sprintf(str, "O");	
		else if ((val < 0.01f) || (val > 9999.99f))	sprintf(str, "%1.1E", val);
		else										sprintf(str, "%4.2f", val);
	}

	drawRectangle(img, pt1, pt2, color, str, fontScale, textProp);

	if (isnan(val)) {
		line(img, pt1,  pt2, CV_RGB(127, 127, 127), 1, CV_AA);
		line(img, Point(pt2.x, pt1.y),  Point(pt1.x, pt2.y), CV_RGB(127, 127, 127), 1, CV_AA);
	}
}

void CMarker::drawRectangle(Mat &img, Point pt1, Point pt2, const Scalar &color, const std::string &str, double fontScale, byte textProp) const 
{
	rectangle(img, pt1, pt2, color, -1);
	
	if (!str.empty()) {
		Size textSize = getTextSize(str, CV_FONT_HERSHEY_SIMPLEX, fontScale, 1, NULL);
		Size rectSize = cvSize(abs(pt2.x - pt1.x), abs(pt2.y - pt1.y));
		
		Point org(MAX(rectSize.width- textSize.width, 6) / 2,  (rectSize.height + textSize.height) / 2);
		if (textProp & TP_LEFT)		org.x = 3;
		if (textProp & TP_RIGHT)	org.x = MAX(rectSize.width - textSize.width - 3, 3);
		if (textProp & TP_TOP)		org.y = textSize.height + 5;
		if (textProp & TP_BOTTOM)	org.y = rectSize.height - 5;
		org += Point(MIN(pt1.x, pt2.x), MIN(pt1.y, pt2.y));

		double intensity = 0;
		for (int i = 0; i < 3; i++) intensity += color.val[i] / 3;
		CvScalar fontColor = (intensity < 128) ? CV_RGB(255, 255, 255) : CV_RGB(0, 0, 0);

		putText(img, str, org, FONT_HERSHEY_SIMPLEX, fontScale, fontColor, 1, CV_AA);		 // TODO: color	
	}
}

// ======================================== Non-Member ========================================

Mat drawDictionary(const Mat &dictionary, double m)
{
	const int		margin = 2;
	const int		nWords = dictionary.rows;
	const int		blockSize = static_cast<int>(sqrt(dictionary.cols));

	int				width = static_cast<int>(sqrt(nWords));
	int				height = nWords / width;
	if (width * height < nWords) width++;

	Mat res(height * (blockSize + margin) + margin, width * (blockSize + margin) + margin, CV_8UC1, cvScalar(0));

	for (int w = 0; w < nWords; w++) {
		Mat word = dictionary.row(w);
		word = 127.5 + m * 127.5 * word.reshape(0, blockSize);

		int y = w / width;
		int x = w % width;

		int y0 = y * (blockSize + margin) + margin;
		int x0 = x * (blockSize + margin) + margin;

		word.convertTo(res(cvRect(x0, y0, blockSize, blockSize)), res.type());
	}

	return res;
}

} }
