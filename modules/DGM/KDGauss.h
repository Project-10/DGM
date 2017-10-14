// K-Dimensional Gauss function 
// Written by Sergey G. Kosov in 2012 - 2014 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	// ================================ NDGauss Class ==============================
	/**
	* @brief Multivariate Gaussian distribution class
	* @details This class allows to approximate the distribution of random variables (samples) \f$ \textbf{x} \f$, represented as \a k - dinemstional
	* points \f$ [x_1,x_2,\dots,x_k] \f$. Under the assumption, that the random variables \f$ \textbf{x} \f$ are normally distributed, the approximation 
	* is done with <a href="http://en.wikipedia.org/wiki/Multivariate_normal_distribution"> multivariate normal distribution</a>:
	* \f[ \mathcal{N}_k(\mu,\Sigma)= \alpha\cdot\exp\big( -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\big), \f] 
	* whith coefficient \f$ \alpha = \frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}} \f$. The parameters \f$ \mu\f$ and \f$\Sigma \f$ of the Gaussian function
	* \f$ \mathcal{N}(\mu,\Sigma) \f$ may be estimated sequentially with function addPoint() or operator operator+=(const Mat &); also they may be set directly with 
	* functions setMu() and setSigma().
	* 
	* The value of the Gaussian function \f$ \mathcal{N}(\mu,\Sigma) \f$ in point \f$ \textbf{x} \f$ is achieved with the functions getAlpha() and getValue():
	* @code
	* double value = CKDGauss::getAlpha() * CKDGauss::getValue(x);
	* @endcode
	*
	* In order to generate a random sample from the distribution, given by Gaussian function \f$ \mathcal{N}(\mu,\Sigma) \f$, one uses function getSample().
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CKDGauss {
	public:
		/**
		* @brief Constructor
		* @param k Dimensions
		*/
		DllExport CKDGauss(dword k);
		/**
		* @brief Constuctor
		* @details The dimension of the Gauss function will be derived from the dimension \a k of the argument \f$\mu\f$
		* @param mu The mean vector \f$\mu\f$ : Mat(size: k x 1; type: CV_XXC1)
		*/
		DllExport CKDGauss(const Mat &mu);
		/**
		* @brief Copy constructor
		*/
		DllExport CKDGauss(const CKDGauss &rhs);

		/**
		* @brief Copy operator
		*/
		DllExport CKDGauss & operator=  (const CKDGauss & rhs);
		/**
		* @brief Merge operator
		* @details This operator merges two Gaussian distributions together:
		* \f[ \begin{aligned}
		* \hat{\mu}    &= \frac{n_1\mu_1 + n_2\mu_2}{n1 + n2} \\
		* \hat{\Sigma} &= \frac{n_1(\Sigma_1 + \mu_1\mu^{\top}_{1}) + n2(\Sigma_2 + \mu_2\mu^{\top}_{2})}{n_1 + n_2} - \hat{\mu}\hat{\mu}^\top \\ 
		* \end{aligned} \f]
		*/
		DllExport CKDGauss & operator+= (const CKDGauss & rhs);
		/**
		* @brief Merge operator
		* @details This operator is equavalent to 
		* @code operator+=(CKDGauss( point )) @endcode
		* and might be used for sequentian estimation of the Gaussian distribution
		*/
		DllExport CKDGauss & operator+= (const Mat &point); 

		/**
		* @brief Clears class variables
		* @details Allows to re-use the class
		*/
		DllExport void			clear(void);
		/**
		* @brief Checks weather the Gaussian function is approximated
		* @retval true if the Gaussian had at least 1 point for approximation, or
		* @retval false otherwise
		*/
		DllExport bool			empty(void) const			 { return (m_nPoints == 0); }

		///@{ 
		/// @name Accessors
		/**
		* @brief Sets the number of approximation points \f$n\f$.
		* @param nPoints The number of sample points, used for the approximation.
		*/
		DllExport void			setNumPoints(size_t nPoints) { m_nPoints = nPoints; }
		/**
		* @brief Returns the number of approximation points \f$n\f$.
		* @return The number of sample points, used for the approximation.
		*/
		DllExport size_t		getNumPoints(void) const { return m_nPoints; }
		/**
		* @brief Sets \f$\mu\f$.
		* @param mu The mean vector \f$\mu\f$ : Mat(size: k x 1; type: CV_64FC1)
		*/
		DllExport void			setMu(const Mat &mu);
		/**
		* @brief Returns \f$\mu\f$.
		* @return The mean vector \f$\mu\f$: Mat(size: k x 1; type: CV_64FC1)
		*/
		DllExport Mat			getMu(void) const { return m_mu.clone(); }				
		/**
		* @brief Sets \f$\Sigma\f$.
		* @param sigma The covariance matrix \f$\Sigma\f$: Diagonal Mat(size: k x k; type: CV_64FC1)
		*/
		DllExport void			setSigma(const Mat &sigma);
		/**
		* @brief Returns \f$\Sigma\f$.
		* @return The covariance matrix \f$\Sigma\f$: Mat(size: k x k; type: CV_64FC1)
		*/
		DllExport Mat			getSigma(void) const { return m_sigma.clone(); }
		///@}

		///@{
		/// @name Main functionality
		/**
		* @brief Adds new k-dimensional point (sample) for Gaussian distribution approximation
		* @details This function is an analog to the merging operator (Ref. operator+=()) and updates the Gaussian 
		* distribution in respect to one new observed point, using the following update rooles:\n
		* \f[ \begin{aligned} 
		* \hat{\mu}    &= \frac{n\mu + p}{n + 1} \\
		* \hat{\Sigma} &= \frac{n(\Sigma + \mu\mu^\top) + p~p^\top}{n + 1} - \hat{\mu}\hat{\mu}^\top \\
		* \end{aligned} \f]
		* @code
		* while(point) estGaussian.addPoint(point);			// estimated Gauss function is updated
		* @endcode
		* @param point k-dimensional point: Mat(size: k x 1; type: CV_64FC1)
		* @param approximate Flag indicating whether a faster approximation of the update rool for \f$\Sigma\f$ should be used:
		* \f$ \hat{\Sigma} = \frac{n\Sigma + (p-\hat{\mu})(p-\hat{\mu})^\top}{n + 1} \f$.
		* For large \f$n\f$ this approximation is equevalent to the original update rool.
		*/
		DllExport void			addPoint(const Mat &point, bool approximate = false);
		DllExport long double	getAlpha(void) const;
		DllExport double		getValue(const Mat &x) const;
		DllExport Mat			getSample(void) const;
		///@}

		/**
		* @brief Returns the Euclidian distance between argument point \b x and the center of multivariate normal distribution \f$\mu\f$.
		* @details The Euclidian distance is calculated by the formula: \f$D_E(\mathcal{N};x)=\sqrt{ (x-\mu)^\top(x-\mu) }\f$.
		* @param x A k-dimensional point (sample): Mat(size: k x 1; type: CV_64FC1)
		* @return The Euclidian distance: \f$D_E(x)\f$
		*/
		DllExport double		getEuclidianDistance(const Mat &x) const;
		/**
		* @brief Returns the Mahalanobis distance between argument point \b x and the center of multivariate normal distribution \f$\mathcal{N}(\mu,\Sigma)\f$.
		* @details The Mahalanobis distance is calculated by the formula: \f$D_M(\mathcal{N};x)=\sqrt{ (x-\mu)^\top\Sigma^{-1}(x-\mu) }\f$
		* @param x n-dimensional point (sample): Mat(size: k x 1; type: CV_64FC1)
		* @return the Mahalanobis distance: \f$D_M(x)\f$
		*/
		DllExport double		getMahalanobisDistance(const Mat &x) const;
		/**
		* @brief Returns the Kullback-Leiber divergence from the multivariate normal distribution \f$\mathcal{N}(\mu,\Sigma)\f$ to argument multivariate normal distribution \f$\mathcal{N}_x(\mu_x,\Sigma_x)\f$.
		* @details The Kullback-Leiber divergence (or relative entropy) is calculated by the formula: \f$D_{KL}(\mathcal{N};\mathcal{N}_x)=\frac{1}{2}\Big( tr(\Sigma^{-1}_{x}\Sigma) +
		* D^{2}_{M}(\mathcal{N}_x;\mu) - k - \ln\big(\frac{\left|\Sigma\right|}{\left|\Sigma_x\right|}\big) \Big)\f$ and expressed in <a href="http://en.wikipedia.org/wiki/Nat_(information)">nats</a>.
		* Here \f$D_M(\mathcal{N}_x;\mu)\f$ is the \a Mahalanobis \a distance beween the centers of multivariate normal distributions \f$\mu_x\f$ and \f$\mu\f$ with respect to \f$\mathcal{N}_x\f$
		* (see getMahalanobisDistance() for more details). Please note, that it is not a symmetrical quantity, that is to say \f$ D_{KL}(\mathcal{N};\mathcal{N}_x) \neq D_{KL}(\mathcal{N}_x;\mathcal{N})\f$
		* and so could be hardly used as a "distance".
		* @param x multivariate normal distribution \f$\mathcal{N}_x(\mu_x,\Sigma_x)\f$ of the dimension \a k.
		* @return the Kullback-Leiber divergence: \f$D_{KL}(\mathcal{N}_x)\f$
		*/
		DllExport double		getKullbackLeiberDivergence(const CKDGauss &x) const;


	protected:
		size_t	m_nPoints;		///< Number of samples
		Mat		m_mu;			///< The mathematical expectation \f$mu\f$: (size: k x 1; type: CV_64FC1)
		Mat		m_sigma;		///< The covariance matrix \f$\Sigma\f$: (size: k x k; type: CV_64FC1)


	private:
		Mat		getSigmaInv(void) const;

	};
}