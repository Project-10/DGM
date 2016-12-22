#include "Global.h"

namespace DirectGraphicalModels { namespace fex { namespace global
{
size_t getNumLines(const Mat &img, int threshold1, int threshold2)
{
	// Converting to one channel image
	Mat I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I); 
		
	GaussianBlur(I, I, Size(5, 5), 0.75, 0.75);		// smooth it, otherwise a lot of false circles may be detected
	
	Mat canny8b;
	Canny(I, canny8b, threshold1 / 2, threshold1, 3);

#if 1
	std::vector<Vec2f> vLines;
	HoughLines(	  canny8b			// image
				, vLines			// lines
				, 1					// distance resolution
				, 5 * CV_PI / 180	// angle resolution
				, threshold2		// lower -> more lines
			);
#else
	std::vector<Vec4i> vLines;
	HoughLinesP(  canny8b			// image
				, vLines			// lines
				, 1					// distance resolution
				, 1 * CV_PI / 180	// angle resolution
				, threshold2		// lower -> more lines
				, 2					// min line length
				, canny8b.rows / 2	// max line length
			);
#endif
	

	if (false) {		// Visualization
		imshow("Canny", canny8b);
		
		Mat tmp;
		if (img.channels() == 1)  cvtColor(img, tmp, CV_GRAY2RGB);
		else img.copyTo(tmp);

#if 1
		for (Vec2f &l : vLines) {
			float rho	= l[0];
			float theta = l[1];
			double a = cos(theta);
			double b = sin(theta);
			double x0 = a * rho;
			double y0 = b * rho;
			Point pt1, pt2;
			pt1.x = cvRound(x0 - 1000 * b);
			pt1.y = cvRound(y0 + 1000 * a);
			pt2.x = cvRound(x0 + 1000 * b);
			pt2.y = cvRound(y0 - 1000 * a);
			line(tmp, pt1, pt2, CV_RGB(255, 0, 0), 1, CV_AA);
		}
#else
		for (Vec4i &l : vLines) 
			line(tmp, Point(l[0], l[1]), Point(l[2], l[3]), CV_RGB(255, 0, 0), 1, CV_AA);
#endif
		imshow("detected lines", tmp);
		waitKey();
	}
		
	return vLines.size();
}

size_t getNumCircles(const Mat &img, int threshold1, int threshold2)
{
	// Converting to one channel image
	Mat I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I);

	GaussianBlur(I, I, Size(9, 9), 2, 2);		// smooth it, otherwise a lot of false circles may be detected

	std::vector<Vec3f> vCircles;
	HoughCircles(I				// image
		, vCircles				// circles
		, CV_HOUGH_GRADIENT		// method
		, 1						// dp 
		, 1						// min distance between centers
		, threshold1			// high treshold of the canny
		, threshold2			// lower -> more circles
	);	

	if (false) {		// Visualization
		Mat canny8b;
		Canny(I, canny8b, threshold1 / 2, threshold1, 3);
		imshow("Canny", canny8b);

		Mat tmp;
		if (img.channels() == 1)  cvtColor(img, tmp, CV_GRAY2RGB);
		else img.copyTo(tmp);

		for (Vec3f &c : vCircles) {
			Point center(cvRound(c[0]), cvRound(c[1]));
			int radius = cvRound(c[2]);
			circle(tmp, center, 3, CV_RGB(0, 255, 0), -1, 8, 0);		// draw the circle center
			circle(tmp, center, radius, CV_RGB(255, 0, 0), 1, 8, 0);		// draw the circle outline
		}
		imshow("circles", tmp);
		waitKey();
	}

	return vCircles.size();
}

float getOpacity(const Mat &img)
{
	int		width	= img.cols;
	int		height	= img.rows;
	float	R		= -1.0f;

	// Converting to one channel image
	Mat I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I);

	float _mean = static_cast<float>(mean(I)[0]);
	float res	= 0.0f;

	for (int y = 0; y < height; y++) {
		byte *pI = I.ptr<byte>(y);
		for (int x = 0; x < width; x++) {
			float dx	= x - 0.5f * width;
			float dy	= y - 0.5f * height;
			float dist	= sqrtf(dx*dx + dy*dy);
			if (R < 0) R = dist;
			float weight = 1.0f - dist / R;
			res += weight * fabs(static_cast<float>(pI[x]) - _mean);
		} // x
	} // y

	return res / (width * height);
}

float getVariance(const Mat &img)
{
	// Converting to one channel image
	Mat I;
	if (img.channels() != 1) cvtColor(img, I, CV_RGB2GRAY);
	else img.copyTo(I);

	Scalar mean, stddev;
	meanStdDev(I, mean, stddev);
	float res = static_cast<float>(stddev[0] * stddev[0]);

	return res;
}

} } }