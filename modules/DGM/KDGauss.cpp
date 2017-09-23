#include "KDGauss.h"
#include "macroses.h"

namespace DirectGraphicalModels {
	// Constructor
	CKDGauss::CKDGauss(dword k) 
		: m_nPoints(0)
		, m_mu(Mat::zeros(k, 1, CV_64FC1))
		, m_sigma(Mat::eye(k, k, CV_64FC1))
	{ }

	// Constructor
	CKDGauss::CKDGauss(const Mat &mu) : m_nPoints(1) {
		int k = mu.rows;
		mu.convertTo(m_mu, CV_64FC1);
		m_sigma = Mat::eye(k, k, CV_64FC1);
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
		long double a = static_cast<long double>(this->m_nPoints) / (this->m_nPoints + rhs.m_nPoints);
		this->m_nPoints += rhs.m_nPoints;
		Mat mu;
		addWeighted(this->m_mu, a, rhs.m_mu, (1.0 - a), 0.0, mu);
		
		Mat t1, t2;
		gemm(this->m_mu, this->m_mu, 1.0, this->m_sigma, 1.0, t1, GEMM_2_T);		// t1 = sigma1 + mu1*mu1^T
		gemm(rhs.m_mu, rhs.m_mu, 1.0, rhs.m_sigma, 1.0, t2, GEMM_2_T);				// t2 = sigma2 + mu2*mu2^T
		addWeighted(t1, a, t2, (1.0 - a), 0.0, this->m_sigma);						// sigma^ = a * t1 + (1-a) * t2;
		this->m_sigma -= mu * mu.t();												// sigma^ = a * [sigma1 + mu1*mu1^T] + (1-a) * [sigma2 + mu2*mu2^T] - mu^ * mu^^T;

		this->m_mu = mu;
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

	void CKDGauss::addPoint(const Mat &point) {
		// Assertions
		DGM_ASSERT_MSG(point.size() == m_mu.size(), "Wrong input point size");
		DGM_ASSERT_MSG(point.type() == m_mu.type(), "Wrong input point type");

		if (m_nPoints == 0) point.copyTo(m_mu);
		else {
			long double a = static_cast<long double>(m_nPoints) / (m_nPoints + 1);
			addWeighted(m_mu, a, point, 1.0 - a, 0.0, m_mu);			// mu = a * mu + (1-a) * point

			for (int y = 0; y < m_sigma.rows; y++) {
				double *pSigma = m_sigma.ptr<double>(y);
				for (int x = 0; x < m_sigma.cols; x++) {
					double cr = (point.at<double>(x, 0) - m_mu.at<double>(x, 0)) * (point.at<double>(y, 0) - m_mu.at<double>(y, 0));
					pSigma[x] = cr + static_cast<double>(a * (pSigma[x] - cr));
				} // x
			} // y

		}
		m_nPoints++;
	}

}
