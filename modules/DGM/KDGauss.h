// K-Dimensional Gauss function 
// Written by Sergey G. Kosov in 2012 - 2014 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	class CKDGauss {
	public:
		CKDGauss(dword k);
		CKDGauss(const Mat &mu);
		CKDGauss(const CKDGauss &rhs);
		~CKDGauss(void) {}

		CKDGauss & operator=  (const CKDGauss & rhs);
		CKDGauss & operator+= (const CKDGauss & rhs);
		CKDGauss & operator+= (const Mat &point); 

		void		clear(void);
		bool		empty(void) const			 { return (m_nPoints == 0); }

		// Accessors
		void		setNumPoints(size_t nPoints) { m_nPoints = nPoints; }
		size_t		getNumPoints(void) const	 { return m_nPoints; }
		void		setMu(const Mat &mu);
		Mat			getMu(void) const			 { return m_mu.clone(); }				
		void		setSigma(const Mat &sigma);
		Mat			getSigma(void) const		 { return m_sigma.clone(); }

		// Main functionality
		void		addPoint(const Mat &point, bool approximate = false);
		Mat			getSigmaInv(void) const;
		long double	getAlpha(void) const;
		double		getValue(const Mat &x) const;
		Mat			getSample(void) const;

		double		getEuclidianDistance(const Mat &x) const;
		double		getMahalanobisDistance(const Mat &x) const;


	private:
		size_t	m_nPoints;		// number of samples
		Mat		m_mu;			// the mathematical expectation (size: k x 1; type: CV_64FC1)
		Mat		m_sigma;		// the covariance matrix (size: k x k; type: CV_64FC1)
	};
}