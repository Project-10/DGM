#include "KDGauss.h"
#include "mathop.h"
#include "macroses.h"

namespace DirectGraphicalModels {
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
