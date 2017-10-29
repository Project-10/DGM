// K-Dimensional Gauss function 
// Written by Sergey G. Kosov in 2012 - 2014 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	// ================================ NDGauss Class ==============================
	/**
	* @brief Multi - dimensional Gauss function class
	* @details This class allows to approximate the distribution of random variables (samples) \f$ \textbf{x} \f$, represented as \a k - dinemstional
	* points \f$ [x_1,x_2,\dots,x_k] \f$. Under the assumption, that the random variables \f$ \textbf{x} \f$ are normally distributed,
	* the approximation is done with <a href="http://en.wikipedia.org/wiki/Multivariate_normal_distribution"> multivariate normal distribution</a>:
	* \f$ \mathcal{N}_k(\mu,\Sigma)= \alpha\cdot\exp\big( -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\big) \f$ whith coefficient
	* \f$ \alpha = \frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}} \f$. The parameters \f$ \mu\f$ and \f$\Sigma \f$ of the Gaussian function
	* \f$ \mathcal{N}(\mu,\Sigma) \f$ may be estimated sequentially with function addPoint() or operator operator+(); also they may be set directly with functions setMu() and setSigma().

	* The value of the Gaussian function \f$ \mathcal{N}(\mu,\Sigma) \f$ in point \f$ \textbf{x} \f$ is achieved with functions getAlpha() and getValue():
	* @code
	* double value = CNDGauss::getAlpha() * CNDGauss::getValue(x);
	* @endcode
	* In order to generate a random sample from the distribution, given by Gaussian function \f$ \mathcal{N}(\mu,\Sigma) \f$, one uses function getSample().
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CKDGauss
	{
	public:
		/**
		* @brief Constructor
		* @param k dimensions
		*/
		DllExport CKDGauss(dword k);
		/// @cond
		/**
		* @brief Constuctor
		* @details The dimension of the Gauss function will be derived from the dimension \a k of the argument \f$\mu\f$
		* @param mu the mean vector \f$\mu\f$ : Mat(size: k x 1; type: CV_64FC1)
		*/
		CKDGauss(const Mat &mu);
		/// @endcond
		DllExport CKDGauss(const CKDGauss &rhs);
		DllExport ~CKDGauss(void);

		DllExport CKDGauss & operator= (const CKDGauss & rhs);
		DllExport CKDGauss & operator+=	(const CKDGauss & rhs);
		/**
		* @brief Merging operator
		* @details This operator merges two Gauss finctions into one: \n\n
		* \f$ \hat{\mu} = \frac{n_1\mu_1 + n_2\mu_2}{n_1 + n_2} \f$;
		* \f$ \hat{\Sigma} = \frac{n_1(\Sigma_1 + \mu_1\mu^{\top}_{1}) + n_2(\Sigma_2 + \mu_2\mu^{\top}_{2})}{n_1 + n_2} - \hat{\mu}\hat{\mu}^\top\f$ \n\n
		* For the sequential Gauss function estimation one also may use this operator (see the code below) or addPoint() function:
		* @code
		* CNDGauss newGaussian(point.rows);		// auxilary Gauss function
		* while(point) {							// set of all sample points
		* newGaussian.setMu(point);			// mu is set to be equal to the sample
		* estGaussian += newGaussian;			// estimated Gauss function is updated
		* }
		* @endcode
		* In this case \f$n_2 = 1\f$ and \f$\Sigma_2 = 0\f$.
		*/
		DllExport const	CKDGauss operator+ (const CKDGauss & rhs) const { return CKDGauss(*this) += rhs; }

		/**
		* @brief Clears class variables
		* @details Allows to re-use the class
		*/
		DllExport void			clear(void);
		/**
		* @brief Freezes the state of the Gaussian function
		* @details This is an optimization function, which calculates and fills internal variables need for major
		* get- accessors of this class. If the function was not called, tese variable will be calculated in the
		* get- accessors every tiime on call.
		*/
		DllExport void			freeze(void);
		/**
		* @brief Adds new n-dimensional point (sample) for Gaussian function approximation
		* @details This function is a faster approximation of the merging operator (Ref. operator+()):\n
		* \f$ \hat{\mu} = \frac{n\mu + p}{n + 1} \f$;
		* \f$ \hat{\Sigma} = \frac{n\Sigma + (p-\hat{\mu})(p-\hat{\mu})^\top}{n + 1}\f$
		* @code
		* while(point) estGaussian.addPoint(point);			// estimated Gauss function is updated
		* @endcode
		* @param point k-dimensional point: Mat(size: k x 1; type: CV_64FC1)
		*/
		DllExport void			addPoint(Mat &point);
		/**
		* @brief Checks weather the Gaussian function is approximated
		* @retval TRUE if the Gaussian had at least 1 point for approximation, or
		* @retval FALSE otherwise
		*/
		DllExport bool			empty(void)	const { return (m_nPoints == 0); }
		/**
		* @brief Sets \f$\mu\f$.
		* @param mu the mean vector \f$\mu\f$ : Mat(size: k x 1; type: CV_64FC1)
		*/
		DllExport void			setMu(Mat &mu);
		/**
		* @brief Returns \f$\mu\f$.
		* @return the mean vector \f$\mu\f$: Mat(size: k x 1; type: CV_64FC1)
		*/
		DllExport Mat			getMu(void) const { return m_mu; }
		/**
		* @brief Sets \f$\Sigma\f$.
		* @param sigma the covariance matrix \f$\Sigma\f$: Diagonal Mat(size: k x k; type: CV_64FC1)
		*/
		DllExport void			setSigma(Mat &sigma);
		/**
		* @brief Returns \f$\Sigma\f$.
		* @return the covariance matrix \f$\Sigma\f$: Mat(size: k x k; type: CV_64FC1)
		*/
		DllExport Mat			getSigma(void) const { return m_sigma; }
		/**
		* @brief Returns \f$\Sigma^{-1}\f$.
		* @return The inversed covariance matrix \f$\Sigma^{-1}\f$: Mat(size: k x k; type: CV_64FC1)
		*/
		DllExport Mat			getSigmaInv(void) const;
		/**
		* @brief Returns \f$\alpha\f$.
		* @return The Gaussian coefficient \f$\alpha\f$.
		*/
		DllExport long double	getAlpha(void) const;
		/**
		* @brief Returns unscaled value of the Gaussian function
		* @details This function returns unscaled value of the Gaussian function, \a i.e. \f$ \exp\big( -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\big) \f$. In order to
		* get the value of \f$ \mathcal{N}_k(\mu,\Sigma) \f$, the output of this function must be multiplied with \f$ \alpha \f$ from the getAlpha() function.
		* @note Three auxilary parameters \b aux1, \b aux2 and \b aux3 are needed for more efficient sequential calculation and do not influent the resulting value, \a e.g.
		* the code:
		* @code
		* Mat aux1, aux2, aux3;
		* for (int i = 0; i < 100; i++) y[i] = getValue(x[i], aux1, aux2, aux3);
		* @endcode
		* will run about two times faster, then the code:
		* @code
		* for (int i = 0; i < 100; i++) y[i] = getValue(x[i]);
		* @endcode
		* > This function is PPL-safe function.
		* @param x n-dimensional point (sample): Mat(size: k x 1; type: CV_64FC1)
		* @param aux1 Auxilary variable
		* @param aux2 Auxilary variable
		* @param aux3 Auxilary variable
		* @return unscaled value of the Gaussian function
		*/
		DllExport double		getValue(Mat &x, Mat &aux1 = Mat(), Mat &aux2 = Mat(), Mat &aux3 = Mat()) const;

		/**
		* @brief Sets the number of approximation points.
		* @param nPoints the number of sample points, used for the approximation.
		*/
		DllExport void			setNumPoints(long nPoints) { m_nPoints = nPoints; }
		/**
		* @brief Returns the number of approximation points.
		* @return the number of sample points, used for the approximation.
		*/
		DllExport size_t		getNumPoints(void) const { return m_nPoints; }
		/**
		* @brief Returns the Euclidian distance between argument point \a x and the center of multivariate normal distribution \f$\mu\f$.
		* @details The Euclidian distance is calculated by the formula: \f$D_E(\mathcal{N};x)=\sqrt{ (x-\mu)^\top(x-\mu) }\f$.
		* @param x n-dimensional point (sample): Mat(size: k x 1; type: CV_64FC1)
		* @return the Euclidian distance: \f$D_E(x)\f$
		*/
		DllExport double		getEuclidianDistance(const Mat &x) const;
		/**
		* @brief Returns the Mahalanobis distance between argument point \a x and the center of multivariate normal distribution \f$\mathcal{N}(\mu,\Sigma)\f$.
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

		/**
		* @brief Returns a random vector (sample) from multivariate normal distribution
		* @details The implementation is based on the paper <a href="ftp://ftp.dca.fee.unicamp.br/pub/docs/vonzuben/ia013_2s09/material_de_apoio/gen_rand_multivar.pdf">Generating Random Vectors from the Multivariate Normal Distribution</a>
		* @return n-dimensional point (sample): Mat(size: k x 1; type: CV_64FC1)
		*/
		DllExport Mat			getSample(void) const;

	private:
		static const bool USE_SAFE_SIGMA;
		static const bool SHOW_OPTIMIZATION_HINTS;

	private:
		size_t		  m_nPoints;		// = 0;		// number of samples
		Mat			  m_mu;				// the mathematical expectation (size: k x 1; type: CV_64FC1)
		Mat			  m_sigma;			// the covariance matrix (size: k x k; type: CV_64FC1)
		Mat			  m_sigmaInv;		// = Mat();	// the inverse to the <sigma> matrix
		Mat			  m_Q;				// = Mat(); // aux Mat for getSample()
		long double	  m_alpha;			// = -1;	// gaussian coefficient

		inline void	  reset_SigmaInv_Q_Alpha(void);
		Mat			  calculateQ(void) const;
	};

	using GaussianMixture = std::vector<CKDGauss>;
}

namespace DirectGraphicalModels_Aux
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

	using GaussianMixture = std::vector<CKDGauss>;
}