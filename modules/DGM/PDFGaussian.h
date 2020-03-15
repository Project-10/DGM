// Gaussian-based Probability Density function class
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "IPDF.h"

namespace DirectGraphicalModels 
{
	// ================================ Histogram PDF Class ==============================
	/**
	 * @brief Gaissian-based PDF class
	 * @details This class approximates PDF via Gaussian functions.
	 * @author Sergey G. Kosov, sergey.kosov@project-10.de
	 */
	class CPDFGaussian : public IPDF
	{
	public:
		DllExport CPDFGaussian(void) : IPDF(), m_mu(0), m_sigma2(0) {}
		DllExport virtual ~CPDFGaussian(void) = default;

		DllExport virtual void		reset(void) override;

		DllExport virtual void		addPoint(Scalar point) override;
		DllExport virtual double	getDensity(Scalar point) override;
		DllExport virtual void 		smooth(unsigned int nIt) override;
		DllExport virtual Scalar	min(void) const override { return Scalar(m_mu - 3 * sqrt(m_sigma2)); }
		DllExport virtual Scalar	max(void) const override { return Scalar(m_mu + 3 * sqrt(m_sigma2)); }


	protected:
		DllExport virtual void		saveFile(FILE *pFile) const override;
		DllExport virtual void		loadFile(FILE *pFile) override;

	
	private:
		double	m_mu;
		double	m_sigma2;		// sigma^2
	};
}
