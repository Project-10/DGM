// Trackball Camera class
// Written by Sergey Kosov in 2016 for Project X
#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace DirectGraphicalModels { namespace vis
{
	// ================================ Trackball Camera  Class ===============================
	/**
	* @brief Trackball camera class
	* @details This class inplements the trackball camera, defined in polar coordinate system and looking at origin point \f$O\f$.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CTrackballCamera
	{
	public:
		/**
		* @brief Constructor
		* @details The camera is defined in polar coordinate system with the parameters \b theta, \b phi and \b radius in point \f$P\f$
		* and looks at the origin \f$O\f$, which is initially situated at the zero point.
		* @param theta The azimuth (or azimuthal angle) measured from the azimuth reference direction to the orthogonal projection of the line segment \f$OP\f$ on the reference plane
		* @param phi The inclination (or polar angle) between the zenith direction and the line segment \f$OP\f$
		* @param radius The radial distance of the camera to origin, \a i.e. the Euclidian distance between \f$O\f$ and \f$P\f$
		*/
		CTrackballCamera(float theta, float phi, float radius)
			: m_initTheta(theta)
			, m_initPhi(phi)
			, m_initRadius(radius)
			, m_theta(theta)
			, m_phi(phi)
			, m_radius(radius)
			, m_up(phi >= 0 ? 1.0f : -1.0f)
			, m_target(0)
			, m_view(glm::mat4(1))
			, m_viewNeedsUpdate(true)
		{ }
		virtual ~CTrackballCamera(void) {}

		/**
		* @brief Resets the camera position
		*/
		void		reset(void);
		/**
		* @brief Returns the view matrix within the camera coordinate
		* @returns The view matrix within the camera coordinate
		*/
		glm::mat4	getViewMatrix(void);


	protected:
		void		rotate(float dTheta, float dPhi);
		void		zoom(float dRadius);
		void		pan(float dx, float dy);

		glm::vec3	getCameraOrintation(void) const { return glm::vec3(sinf(m_phi) * sinf(m_theta), cosf(m_phi), sinf(m_phi) * cosf(m_theta)); }
		glm::vec3	getCameraPosition(void) const { return m_target + getCameraOrintation() * m_radius; }
		void		updateViewMatrix(void) { m_view = glm::lookAt(getCameraPosition(), m_target, glm::vec3(0, m_up, 0)); }


	private:
		float		m_initTheta;
		float		m_initPhi;
		float		m_initRadius;
		float		m_theta;
		float		m_phi;
		float		m_radius;
		float		m_up;

		glm::vec3	m_target;
		glm::mat4	m_view;

		bool		m_viewNeedsUpdate;
	};

} }
