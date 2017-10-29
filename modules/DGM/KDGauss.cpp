#include "KDGauss.h"
#include "random.h"
#include "mathop.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	// Constants
	const bool CKDGauss::USE_SAFE_SIGMA = false;
	const bool CKDGauss::SHOW_OPTIMIZATION_HINTS = false;

	// Constructor
	CKDGauss::CKDGauss(dword k) : m_nPoints(0), m_sigmaInv(Mat()), m_Q(Mat()), m_alpha(-1.0)
	{
		m_mu = Mat(k, 1, CV_64FC1, Scalar(0));
		m_sigma = USE_SAFE_SIGMA ? Mat::eye(k, k, CV_64FC1) : Mat(k, k, CV_64FC1, Scalar(0));
	}

	/// @cond 
	// Constructor for internal use only
	CKDGauss::CKDGauss(const Mat &mu) : m_nPoints(1), m_sigmaInv(Mat()), m_Q(Mat()), m_alpha(-1.0)
	{
		dword k = mu.rows;
		mu.convertTo(m_mu, CV_64FC1);
		m_sigma = USE_SAFE_SIGMA ? Mat::eye(k, k, CV_64FC1) : Mat(k, k, CV_64FC1, Scalar(0));
	}
	/// @endcond

	// Copy Constructor
	CKDGauss::CKDGauss(const  CKDGauss &rhs) : m_nPoints(rhs.m_nPoints), m_alpha(rhs.m_alpha)
	{
		rhs.m_mu.copyTo(this->m_mu);
		rhs.m_sigma.copyTo(this->m_sigma);
		if (rhs.m_sigmaInv.empty()) this->m_sigmaInv = Mat();
		else rhs.m_sigmaInv.copyTo(this->m_sigmaInv);
		if (rhs.m_Q.empty()) this->m_Q = Mat();
		else rhs.m_Q.copyTo(this->m_Q);
	}

	// Destructor
	CKDGauss::~CKDGauss(void)
	{
		m_mu.release();
		m_sigma.release();
		if (!m_sigmaInv.empty()) m_sigmaInv.release();
		if (!m_Q.empty()) m_Q.release();
	};

	// Copy Operator
	CKDGauss & CKDGauss::operator=(const CKDGauss &rhs)
	{
		if (this != &rhs) {
			this->m_nPoints = rhs.m_nPoints;
			rhs.m_mu.copyTo(this->m_mu);
			rhs.m_sigma.copyTo(this->m_sigma);
			if (rhs.m_sigmaInv.empty()) {
				if (!this->m_sigmaInv.empty()) this->m_sigmaInv.release();
			}
			else rhs.m_sigmaInv.copyTo(this->m_sigmaInv);
			if (rhs.m_Q.empty()) {
				if (!this->m_Q.empty()) this->m_Q.release();
			}
			else rhs.m_Q.copyTo(this->m_Q);
			this->m_alpha = rhs.m_alpha;
		}
		return *this;
	}

	// Compound Plus Operator
	CKDGauss & CKDGauss::operator+=(const CKDGauss &rhs)
	{
		long double a = static_cast<long double>(this->m_nPoints) / (this->m_nPoints + rhs.m_nPoints);
		this->m_nPoints += rhs.m_nPoints;
		Mat mu;																		// mu^
		addWeighted(this->m_mu, a, rhs.m_mu, (1.0 - a), 0.0, mu);					// mu^ = a * mu1 + (1-a) * mu2;			

		Mat t1, t2;
		gemm(this->m_mu, this->m_mu, 1.0, this->m_sigma, 1.0, t1, GEMM_2_T);		// t1 = sigma1 + mu1*mu1^T
		gemm(rhs.m_mu, rhs.m_mu, 1.0, rhs.m_sigma, 1.0, t2, GEMM_2_T);		// t2 = sigma2 + mu2*mu2^T
		addWeighted(t1, a, t2, (1.0 - a), 0.0, this->m_sigma);						// sigma^ = a * t1 + (1-a) * t2;
		this->m_sigma -= mu * mu.t();												// sigma^ = a * [sigma1 + mu1*mu1^T] + (1-a) * [sigma2 + mu2*mu2^T] - mu^ * mu^^T;

		mu.copyTo(this->m_mu);
		mu.release();
		return *this;
	}

	void CKDGauss::clear(void)
	{
		m_nPoints = 0;
		m_mu.setTo(0);
		m_sigma.setTo(0);
		reset_SigmaInv_Q_Alpha();
	}

	void CKDGauss::freeze(void)
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

	void CKDGauss::addPoint(Mat &point)
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
					double cr = (point.at<double>(x, 0) - m_mu.at<double>(x, 0)) * (point.at<double>(y, 0) - m_mu.at<double>(y, 0));
					pSigma[x] = cr + static_cast<double>(a * (pSigma[x] - cr));
				} // j
			} // i
		}
		m_nPoints++;
		reset_SigmaInv_Q_Alpha();
	}

	void CKDGauss::setMu(Mat &mu)
	{
		// Assertions
		DGM_ASSERT_MSG(mu.size() == m_mu.size(), "Wrong mu size");
		DGM_ASSERT_MSG(mu.type() == m_mu.type(), "Wrong mu type");

		mu.copyTo(m_mu);
		m_nPoints = 1;
		reset_SigmaInv_Q_Alpha();
	}

	void CKDGauss::setSigma(Mat &sigma)
	{
		// Assertions
		DGM_ASSERT_MSG(sigma.size() == m_sigma.size(), "Wrong sigma size");
		DGM_ASSERT_MSG(sigma.type() == m_sigma.type(), "Wrong sigma type");

		sigma.copyTo(m_sigma);
		m_nPoints = 1;
		reset_SigmaInv_Q_Alpha();
	}

	Mat CKDGauss::getSigmaInv(void) const
	{
		if (m_sigmaInv.empty()) {
			DGM_IF_WARNING(SHOW_OPTIMIZATION_HINTS, "Use CKDGauss::freeze() method in order to pre-calculate this value and speed up sequential calculations");
			Mat sigmaInv;
			invert(m_sigma, sigmaInv, DECOMP_SVD);	// sigmaInv = sigma^-1
			return sigmaInv;
		}
		else return m_sigmaInv;
	}

	long double CKDGauss::getAlpha(void) const
	{
		if (m_alpha < 0) {
			DGM_IF_WARNING(SHOW_OPTIMIZATION_HINTS, "Use CKDGauss::freeze() method in order to pre-calculate this value and speed up sequential calculations");
			int D = m_sigma.cols;
			long double det = MAX(DBL_EPSILON, sqrtl(static_cast<long double>(determinant(m_sigma))));
			long double sPi = powl(2 * static_cast<long double>(Pi), static_cast<long double>(D) / 2);
			long double alpha = 1 / (det * sPi);
			return alpha;
		}
		else return m_alpha;
	}

	double CKDGauss::getValue(Mat &x, Mat &X, Mat &p1, Mat &p2) const
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

	double CKDGauss::getEuclidianDistance(const Mat &x) const
	{
		// Assertions (mathop::Euclidian also checks that)
		DGM_ASSERT_MSG(x.size() == m_mu.size(), "Wrong x size");
		DGM_ASSERT_MSG(x.type() == m_mu.type(), "Wrong x type");

		return mathop::Euclidian<double, double>(m_mu, x);
	}

	double CKDGauss::getMahalanobisDistance(const Mat &x) const
	{
		// Assertions
		DGM_ASSERT_MSG(x.size() == m_mu.size(), "Wrong x size");
		DGM_ASSERT_MSG(x.type() == m_mu.type(), "Wrong x type");

		return static_cast<double>(Mahalanobis(x, m_mu, getSigmaInv()));
	}

	double CKDGauss::getKullbackLeiberDivergence(const CKDGauss &x) const
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

	Mat CKDGauss::getSample(void) const
	{
		Mat X = random::N(m_mu.size(), m_mu.type());				// X - vector of independ random variable with normal distribution

		DGM_IF_WARNING(SHOW_OPTIMIZATION_HINTS && m_Q.empty(), "Use CKDGauss::freeze() method in order to pre-calculate m_Q and speed up sequential calculations");
		Mat Q = m_Q.empty() ? calculateQ() : m_Q;

		Mat res;
		gemm(Q, X, 1, m_mu, 1, res, GEMM_1_T);
		return res;
	}

	// ---------------------- Private functions ----------------------

	inline void CKDGauss::reset_SigmaInv_Q_Alpha(void)
	{
		if (!m_sigmaInv.empty()) m_sigmaInv.release();
		if (!m_Q.empty()) m_Q.release();
		m_alpha = -1.0;
	}

	Mat CKDGauss::calculateQ(void) const
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

namespace DirectGraphicalModels_Aux {
	using namespace DirectGraphicalModels;
	
	// Constructor
	CKDGauss::CKDGauss(dword k) 
		: m_nPoints(0)
		, m_mu(Mat::zeros(k, 1, CV_64FC1))
		, m_sigma(Mat::zeros(k, k, CV_64FC1))
	{ }

	// Constructor
	CKDGauss::CKDGauss(const Mat &mu) : m_nPoints(1) {
		int k = mu.rows;
		mu.convertTo(m_mu, CV_64FC1);
		m_sigma = Mat::zeros(k, k, CV_64FC1);
	}
	
	// Copy Constructor
	CKDGauss::CKDGauss(const CKDGauss &rhs)
		: m_nPoints(rhs.m_nPoints)
		, m_mu(rhs.m_mu.clone())
		, m_sigma(rhs.m_sigma.clone())
	{ }

	// Copy Operator
	CKDGauss & CKDGauss::operator= (const CKDGauss & rhs) {
		if (this != &rhs) {
			this->m_nPoints	= rhs.m_nPoints;
			this->m_mu		= rhs.m_mu.clone();
			this->m_sigma	= rhs.m_sigma.clone();
		}
		return *this;
	}

	CKDGauss & CKDGauss::operator+= (const CKDGauss & rhs) {
		if (rhs.m_nPoints == 1) this->addPoint(rhs.m_mu);
		else {
			long double a = static_cast<long double>(this->m_nPoints) / (this->m_nPoints + rhs.m_nPoints);
			this->m_nPoints += rhs.m_nPoints;

			// fast matrix multiplation (faster than OpenCV gemm())
			// sigma^ = a * (sigma1 + mu1 * mu1^t) + (1 - a) * (sigma2 + mu2 * mu2^t)
			for (int y = 0; y < this->m_sigma.rows; y++) {
				double		 *pThisSigma = this->m_sigma.ptr<double>(y);
				const double *pRhsSigma = rhs.m_sigma.ptr<double>(y);
				for (int x = 0; x < this->m_sigma.cols; x++)
					pThisSigma[x] = static_cast<double>(a * (pThisSigma[x] + this->m_mu.at<double>(x, 0) * this->m_mu.at<double>(y, 0)) +
					(1.0 - a) * (pRhsSigma[x] + rhs.m_mu.at<double>(x, 0) * rhs.m_mu.at<double>(y, 0)));
			} // y
			addWeighted(this->m_mu, a, rhs.m_mu, 1.0 - a, 0.0, this->m_mu);			// mu^ = a * mu1 + (1- a) * mu2
			this->m_sigma -= this->m_mu * this->m_mu.t();							// sigma^ = a * (sigma1 + mu1 * mu1^t) + (1 - a) * (sigma2 + mu2 * mu2^t) - mu^ * mu^^T 
		}
		return *this;
	}

	CKDGauss & CKDGauss::operator+= (const Mat &point) { 
		// return this->operator+=(CKDGauss(point)); 
		addPoint(point);
		return *this;
	}

	void CKDGauss::clear(void) {
		m_nPoints = 0;
		m_mu.setTo(0);
		m_sigma = Mat::eye(m_sigma.size(), m_sigma.type());
	}

	double CKDGauss::getValue(const Mat &x) const {
		// Assertions
		DGM_ASSERT_MSG(x.size() == m_mu.size(), "Wrong x size");
		DGM_ASSERT_MSG(x.type() == m_mu.type(), "Wrong x type");

		Mat X, p1, p2, sigmaInv;
		subtract(x, m_mu, X);							// X = x - mu
		invert(m_sigma, sigmaInv, DECOMP_SVD);			// sigmaInv = sigma^-1
		gemm(sigmaInv, X, 1.0, Mat(), 0.0, p1, 0);
		gemm(X, p1, 1.0, Mat(), 0.0, p2, GEMM_1_T);
		double value = -0.5 * p2.at<double>(0, 0);		// val = -0.5 * (X-mu)^T * Sigma^-1 * (X-mu)

		return exp(value);
	}

	void CKDGauss::addPoint(const Mat &point, bool approximate) {
		// Assertions
		DGM_ASSERT_MSG(point.size() == m_mu.size(), "Wrong input point size");
		DGM_ASSERT_MSG(point.type() == m_mu.type(), "Wrong input point type");

		if (m_nPoints == 0) point.copyTo(m_mu);
		else {
			long double a = static_cast<long double>(m_nPoints) / (m_nPoints + 1);

			if (approximate) { // ---------------- approximate calculation of sigma ----------------
				addWeighted(m_mu, a, point, 1.0 - a, 0.0, m_mu);		// mu = a * mu + (1-a) * point

				// fast matrix multiplation (faster than OpenCV gemm())
				// sigma^ = a * sigma1 + (1 - a) * (point * point^t - mu1 * mu1^T) 
				for (int y = 0; y < m_sigma.rows; y++) {
					double *pSigma = m_sigma.ptr<double>(y);
					for (int x = 0; x < m_sigma.cols; x++) {
						double cr = (point.at<double>(x, 0) - m_mu.at<double>(x, 0)) * (point.at<double>(y, 0) - m_mu.at<double>(y, 0));
						pSigma[x] = cr + static_cast<double>(a * (pSigma[x] - cr));
					} // x
				} // y
			} 
			else { // ---------------- general exact calculation of sigma ----------------
				Mat mu;
				addWeighted(this->m_mu, a, point, 1.0 - a, 0.0, mu);	// mu^ = a * mu1 + (1- a) * point

				// fast matrix multiplation (faster than OpenCV gemm())
				// sigma^ = a * (sigma1 + mu1 * mu1^t) + (1 - a) * point * point^t - mu^ * mu^^T 
				for (int y = 0; y < m_sigma.rows; y++) {
					double *pSigma = m_sigma.ptr<double>(y);
					for (int x = 0; x < m_sigma.cols; x++) {
						pSigma[x] = static_cast<double>(a * (pSigma[x] + m_mu.at<double>(x, 0) * m_mu.at<double>(y, 0)) +
							(1.0 - a) *  point.at<double>(x, 0) * point.at<double>(y, 0) -
							mu.at<double>(x, 0) * mu.at<double>(y, 0));
					} // x
				} // y
				this->m_mu = mu;
			}
		}
		m_nPoints++;
	}

	long double CKDGauss::getAlpha(void) const {
		int k = m_sigma.cols;
		long double det = MAX(DBL_EPSILON, sqrtl(static_cast<long double>(determinant(m_sigma))));
		long double sPi = powl(2 * static_cast<long double>(Pi), static_cast<long double>(k) / 2);
		long double alpha = 1 / (det * sPi);
		return alpha;
	}

	double CKDGauss::getEuclidianDistance(const Mat &x) const
	{
		return mathop::Euclidian<double, double>(m_mu, x);
	}

	double CKDGauss::getMahalanobisDistance(const Mat &x) const
	{
		return Mahalanobis(m_mu, x, getSigmaInv());
	}

	double CKDGauss::getKullbackLeiberDivergence(const CKDGauss &x) const
	{
		// Assertions
		DGM_ASSERT_MSG(x.getMu().size() == m_mu.size(), "Wrong x.mu size");
		DGM_ASSERT_MSG(x.getMu().type() == m_mu.type(), "Wrong x.mu type");

		Mat p;
		gemm(x.getSigmaInv(), m_sigma, 1.0, Mat(), 0.0, p, 0);		// p = \Sigma^{-1}_{x} * \Sigma
		Scalar tr = trace(p);
		p.release();

		double dst = x.getMahalanobisDistance(m_mu);

		int k = m_mu.rows;

		double ln = log(determinant(m_sigma) / determinant(x.getSigma()));

		double res = static_cast<double>(tr.val[0] + dst*dst - k - ln) / 2;

		return res;
	}

	// =============================== Private ===============================
	Mat	CKDGauss::getSigmaInv(void) const
	{
		Mat sigmaInv;
		invert(m_sigma, sigmaInv, DECOMP_SVD);	// sigmaInv = sigma^-1
		return sigmaInv;
	}
}
