#include "Random.h"
#include <time.h>

namespace DirectGraphicalModels
{
// Constructor
CRandom::CRandom(void)
{
	if (!isInitialized) {
		std::srand(static_cast<unsigned int>(time(NULL)));	
		//std::srand(0);
		isInitialized = true;
	}
}

dword CRandom::du(void) const
{
	dword r1 = rand();
	dword r2 = rand();
	return r1 * (RAND_MAX + 1) + r2;
}

Mat CRandom::U(dword k) const
{
	Mat res(k, 1, CV_32FC1);
	for (dword i = 0; i < k; i++) res.at<float>(i, 0) = U();
	return res;
}

float CRandom::U(float a, float b) const
{
	if (a < b)	return a + (b - a) * U();
	else		return b + (a - b) * U();
}

Mat CRandom::U(dword k, float a, float b) const
{
	Mat res(k, 1, CV_32FC1);
	for (dword i = 0; i < k; i++) res.at<float>(i, 0) = U(a, b);
	return res;
}

float CRandom::N(void) const
{
	float res = 0;
	for (int i = 0; i < NUM_SAMPLES; i++) res += U();
	res = (res - static_cast<float>(NUM_SAMPLES)/2) / sqrtf(static_cast<float>(NUM_SAMPLES)/12);
	return res;
}

Mat CRandom::N(dword k) const
{
	Mat res(k, 1, CV_32FC1);
	for (dword i = 0; i < k; i++) res.at<float>(i, 0) = N();
	return res;
}

Mat	CRandom::N(dword k, float mu, float sigma) const
{
	Mat res(k, 1, CV_32FC1);
	for (dword i = 0; i < k; i++) res.at<float>(i, 0) = N(mu, sigma);
	return res;
}
}
