// Gaussian-based Probability Density function class
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "PDF.h"

namespace DirectGraphicalModels 
{
	// ================================ Histogram PDF Class ==============================
	/**
	* @brief Gaissian-based PDF class
	* @details This class approximates PDF via Gaussian functions. 
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CPDFGaussian : public CPDF
	{
	public:
		DllExport CPDFGaussian(void) : CPDF(), m_mu(0), m_sigma2(0) {}
		DllExport virtual ~CPDFGaussian(void) {}

		DllExport virtual void	reset(void);

		DllExport virtual void	addPoint(float point);
		DllExport virtual float	getDensity(float point); 
		DllExport virtual float min(void) const { return m_mu - 3 * sqrtf(m_sigma2); }
		DllExport virtual float max(void) const { return m_mu + 3 * sqrtf(m_sigma2); }


	protected:
		DllExport virtual void	saveFile(FILE *pFile) const;
		DllExport virtual void	loadFile(FILE *pFile);

	
	private:
		float	m_mu;
		float	m_sigma2;		// sigma^2
	};
}
