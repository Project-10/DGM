#include "NDGauss.h"
#include "random.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constants
const bool CNDGauss::USE_SAFE_SIGMA = true;
const bool CNDGauss::SHOW_OPTIMIZATION_HINTS = false;

// Constructor
CNDGauss::CNDGauss(dword k) : m_nPoints(0), m_sigmaInv(Mat()), m_Q(Mat()), m_alpha(-1.0)
{
	m_mu	= Mat(k, 1, CV_64FC1, Scalar(0));
	m_sigma = USE_SAFE_SIGMA ? Mat::eye(k, k, CV_64FC1) : Mat(k, k, CV_64FC1, Scalar(0));
}

/// @cond 
// Constructor for internal use only
CNDGauss::CNDGauss(const Mat &mu) : m_nPoints(1), m_sigmaInv(Mat()), m_Q(Mat()), m_alpha(-1.0)
{
	dword k = mu.rows;
	mu.convertTo(m_mu, CV_64FC1);
	m_sigma = USE_SAFE_SIGMA ? Mat::eye(k, k, CV_64FC1) : Mat(k, k, CV_64FC1, Scalar(0));
}
/// @endcond

// Copy Constructor
CNDGauss::CNDGauss(const  CNDGauss &rhs) : m_nPoints(rhs.m_nPoints), m_alpha(rhs.m_alpha) 
{
	rhs.m_mu.copyTo(this->m_mu);
	rhs.m_sigma.copyTo(this->m_sigma);		
	if (rhs.m_sigmaInv.empty()) this->m_sigmaInv = Mat();
	else rhs.m_sigmaInv.copyTo(this->m_sigmaInv);
	if (rhs.m_Q.empty()) this->m_Q = Mat();
	else rhs.m_Q.copyTo(this->m_Q);
}

// Destructor
CNDGauss::~CNDGauss(void) 
{
	m_mu.release();
	m_sigma.release();
	if (!m_sigmaInv.empty()) m_sigmaInv.release();
	if (!m_Q.empty()) m_Q.release();
};

// Copy Operator
CNDGauss & CNDGauss::operator=(const CNDGauss &rhs)
{
    if (this != &rhs) {
		this->m_nPoints = rhs.m_nPoints;
		rhs.m_mu.copyTo(this->m_mu);
		rhs.m_sigma.copyTo(this->m_sigma);
		if (rhs.m_sigmaInv.empty()) {
			if (!this->m_sigmaInv.empty()) this->m_sigmaInv.release();
		} else rhs.m_sigmaInv.copyTo(this->m_sigmaInv);
		if (rhs.m_Q.empty()) {
			if (!this->m_Q.empty()) this->m_Q.release();
		}else rhs.m_Q.copyTo(this->m_Q);
		this->m_alpha = rhs.m_alpha;
	}
	return *this;
}

// Compound Plus Operator
CNDGauss & CNDGauss::operator+=(const CNDGauss &rhs)
{
	long double a = static_cast<long double>(this->m_nPoints) / (this->m_nPoints + rhs.m_nPoints);
	this->m_nPoints += rhs.m_nPoints;	
	Mat mu;																		// mu^
	addWeighted(this->m_mu, a, rhs.m_mu, (1.0 - a), 0.0, mu);					// mu^ = a * mu1 + (1-a) * mu2;			

	Mat t1, t2;
	gemm(this->m_mu, this->m_mu, 1.0, this->m_sigma, 1.0, t1, GEMM_2_T);		// t1 = sigma1 + mu1*mu1^T
	gemm(rhs.m_mu,   rhs.m_mu,   1.0, rhs.m_sigma,   1.0, t2, GEMM_2_T);		// t2 = sigma2 + mu2*mu2^T
	addWeighted(t1, a, t2, (1.0 - a), 0.0, this->m_sigma);						// sigma^ = a * t1 + (1-a) * t2;
	this->m_sigma -= mu * mu.t();												// sigma^ = a * [sigma1 + mu1*mu1^T] + (1-a) * [sigma2 + mu2*mu2^T] - mu^ * mu^^T;

	mu.copyTo(this->m_mu);
	mu.release();
	return *this;
}

void CNDGauss::clear(void)
{
	m_nPoints = 0;
	m_mu.setTo(0);
	m_sigma.setTo(0);
	reset_SigmaInv_Q_Alpha();
}

void CNDGauss::freeze(void)
{
	// m_sigmaInv
	if (m_sigmaInv.empty()) 
		invert(m_sigma, m_sigmaInv, DECOMP_SVD);

	// m_Q
	if (m_Q.empty()) m_Q = calculateQ();

	// m_alpha
	if (m_alpha < 0) {
		int D = m_sigma.cols;
		long double det = MAX(LDBL_EPSILON, sqrtl(static_cast<long double>(determinant(m_sigma))));
		long double sPi = powl(2 * static_cast<long double>(Pi), static_cast<long double>(D) / 2);
		m_alpha = 1 / (det * sPi);
	}
}

void CNDGauss::addPoint(Mat &point)
{
	// Assertions
	DGM_ASSERT_MSG(point.size() == m_mu.size(), "Wrong input point size");
	DGM_ASSERT_MSG(point.type() == m_mu.type(), "Wrong input point type");

 	if (m_nPoints == 0) point.copyTo(m_mu);
	else {
		long double a = static_cast<long double>(m_nPoints) / (m_nPoints + 1.0);
		addWeighted(m_mu, a, point, (1.0 - a), 0.0, m_mu);				// mu = a * mu + (1-a) * point

		int D = m_sigma.cols;											// dimension
		for (int y = 0; y < D; y++) {
			double *pSigma = m_sigma.ptr<double>(y);
			for (int x = 0; x < D; x++) {
				double cr = (point.at<double>(x, 0) - m_mu.at<double>(x, 0)) * (point.at<double>(y, 0) - m_mu.at<double>(y,0));
				pSigma[x] = cr + static_cast<double>(a * (pSigma[x] - cr));
			} // j
		} // i
	}
	m_nPoints++;
	reset_SigmaInv_Q_Alpha();
}

void CNDGauss::setMu(Mat &mu)
{
	// Assertions
	DGM_ASSERT_MSG(mu.size() == m_mu.size(), "Wrong mu size");
	DGM_ASSERT_MSG(mu.type() == m_mu.type(), "Wrong mu type");

	mu.copyTo(m_mu);
	m_nPoints = 1;
	reset_SigmaInv_Q_Alpha();
}

void CNDGauss::setSigma(Mat &sigma)
{
	// Assertions
	DGM_ASSERT_MSG(sigma.size() == m_sigma.size(), "Wrong sigma size");
	DGM_ASSERT_MSG(sigma.type() == m_sigma.type(), "Wrong sigma type");

	sigma.copyTo(m_sigma);	
	m_nPoints = 1;
	reset_SigmaInv_Q_Alpha();
}

Mat CNDGauss::getSigmaInv(void) const
{
	if (m_sigmaInv.empty()) {
		DGM_IF_WARNING(SHOW_OPTIMIZATION_HINTS, "Use CNDGauss::freeze() method in order to pre-calculate this value and speed up sequential calculations");
		Mat sigmaInv;
		invert(m_sigma, sigmaInv, DECOMP_SVD);	// sigmaInv = sigma^-1
		return sigmaInv;
	} else return m_sigmaInv;
}

long double CNDGauss::getAlpha(void) const
{
	if (m_alpha < 0) {
		DGM_IF_WARNING(SHOW_OPTIMIZATION_HINTS, "Use CNDGauss::freeze() method in order to pre-calculate this value and speed up sequential calculations");
		int D = m_sigma.cols;
		long double det = MAX(DBL_EPSILON, sqrtl(static_cast<long double>(determinant(m_sigma))));
		long double sPi = powl(2 * static_cast<long double>(Pi), static_cast<long double>(D) / 2);
		long double alpha = 1 / (det * sPi);
		return alpha;
	} else return m_alpha;
}

double CNDGauss::getValue(Mat &x, Mat &X, Mat &p1, Mat &p2) const
{
	// Assertions
	DGM_ASSERT_MSG(x.size() == m_mu.size(), "Wrong x size");
	DGM_ASSERT_MSG(x.type() == m_mu.type(), "Wrong x type");

	subtract(x, m_mu, X);							// X = x - mu
	gemm(getSigmaInv(), X, 1.0, Mat(), 0.0, p1, 0);		
	gemm(X, p1, 1.0, Mat(), 0.0, p2, GEMM_1_T);
	double value = -0.5 * p2.at<double>(0, 0);		// val = -0.5 * (X-mu)^T * Sigma^-1 * (X-mu)

	return exp(value);
}

double CNDGauss::getEuclidianDistance(const Mat &x) const
{
	// Assertions
	DGM_ASSERT_MSG(x.size() == m_mu.size(), "Wrong x size");
	DGM_ASSERT_MSG(x.type() == m_mu.type(), "Wrong x type");
	
	double res = 0.0f;
	int D = m_mu.rows;											// dimension
	for (register int j = 0; j < D; j++)
		res += (x.at<double>(j, 0) - m_mu.at<double>(j,0)) * (x.at<double>(j, 0) - m_mu.at<double>(j,0));

	res = sqrt(res);
	return res;
}

double CNDGauss::getMahalanobisDistance(const Mat &x) const
{
	// Assertions
	DGM_ASSERT_MSG(x.size() == m_mu.size(), "Wrong x size");
	DGM_ASSERT_MSG(x.type() == m_mu.type(), "Wrong x type");
	
 	return static_cast<double>(Mahalanobis(x, m_mu, getSigmaInv()));
}

double CNDGauss::getKullbackLeiberDivergence(CNDGauss &x) const
{
	// Assertions
	DGM_ASSERT_MSG(x.getMu().size() == m_mu.size(), "Wrong x.mu size");
	DGM_ASSERT_MSG(x.getMu().type() == m_mu.type(), "Wrong x.mu type");

	Mat p;
	gemm(x.getSigmaInv(), m_sigma, 1.0, Mat(), 0.0, p, 0);		// p = \Sigma^{-1}_{x} * \Sigma
	CvScalar tr = trace(p);
	p.release();

	double dst = x.getMahalanobisDistance(m_mu);
	
	int k = m_mu.rows;	

	double ln = log(determinant(m_sigma) / determinant(x.getSigma()));
	
	double res = static_cast<double>(tr.val[0] + dst*dst - k - ln) / 2;

	return res;
}

Mat CNDGauss::getSample(void) const
{
	Mat X = random::N(m_mu.size(), 0, 1);				// X - vector of independ random variable with normal distribution

	DGM_IF_WARNING(SHOW_OPTIMIZATION_HINTS && m_Q.empty(), "Use CNDGauss::freeze() method in order to pre-calculate m_Q and speed up sequential calculations");
	Mat Q = m_Q.empty() ? calculateQ() : m_Q;
	
	Mat res;
	gemm(m_Q, X, 1, m_mu, 1, res, GEMM_1_T);		

	return res;
}

// ---------------------- Private functions ----------------------

inline void CNDGauss::reset_SigmaInv_Q_Alpha(void)
{
	if (!m_sigmaInv.empty()) m_sigmaInv.release();
	if (!m_Q.empty()) m_Q.release();
	m_alpha = -1.0;
}

Mat CNDGauss::calculateQ(void) const
{
	int D = m_mu.rows;	// dimension
		
	// eigenvalues and eigenvectors of <sigma>
	Mat E, F;												
	eigen(m_sigma, E, F);

	// Diagonal Matrix L 
	Mat L(D, D, CV_64FC1); L.setTo(0);
	for (int d = 0; d < D; d++) L.at<double>(d, d) = E.at<double>(d, 0);
	sqrt(L, L);			

	// Matrix Q
	Mat res;
	gemm(L, F, 1, Mat(), 0, res);

	return res;
}

}