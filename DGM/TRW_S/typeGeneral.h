#pragma once

#include <string.h>
#include <assert.h>


/// stores edge information and either forward or backward message
class Edge
{
public: 
	void		Initialize(int nStates, double *data);				// called once when user adds an edge
	void		Swap(void)			{ m_dir = 1 - m_dir; } 		// if the client calls this function, then the meaning of 'dir' in distance transform functions is swapped

	// When UpdateMessage() is called, edge contains message from dest to source.
	// The function must replace it with the message from source to dest.
	// The update rule is given below assuming that source corresponds to tail (i) and dest corresponds
	// to head (j) (which is the case if dir==0).
	//
	// 1. Compute Di[ki] = gamma*source[ki] - message[ki].  (Note: message = message from j to i).
	// 2. Compute distance transform: set
	//       message[kj] = min_{ki} (Di[ki] + V(ki,kj)). (Note: message = message from i to j).
	// 3. Compute vMin = min_{kj} m_message[kj].
	// 4. Set m_message[kj] -= vMin.
	// 5. Return vMin.
	//
	// If dir==1 then source corresponds to j, sink corresponds to i. Then the update rule is
	//
	// 1. Compute Dj[kj] = gamma*source[kj] - message[kj].  (Note: message = message from i to j).
	// 2. Compute distance transform: set
	//       message[ki] = min_{kj} (Dj[kj] + V(ki,kj)). (Note: message = message from j to i).
	// 3. Compute vMin = min_{ki} m_message[ki].
	// 4. Set m_message[ki] -= vMin.
	// 5. Return vMin.
	//
	// If Edge::Swap() has been called odd number of times, then the meaning of dir is swapped.
	//
	// Vector 'source' must not be modified. Function may use 'buf' as a temporary storage.
	double UpdateMessage(int nStates, double *source, double gamma, int dir, double *buf);

	// If dir==0, then sets dest[kj] += V(ksource, kj).
	// If dir==1, then sets dest[ki] += V(ki, ksource).
	// If Swap() has been called odd number of times, then the meaning of dir is swapped.
	void AddColumn(int nStates, int ksource, double *dest, int dir);


	int		  m_dir;				// 0 if Swap() was called even number of times, 1 otherwise
	double	* m_data;				// array of size K * K
	double	* m_message;			// array of size K
};
//////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Implementation ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

inline void Edge::Initialize(int nStates, double *data)
{
	m_dir = 0;
	m_data = new double[nStates * nStates];
	memcpy(m_data, data, nStates * nStates * sizeof(double));

	m_message = new double[nStates];
	memset(m_message, 0, nStates * sizeof(double));
}

inline double Edge::UpdateMessage(int nStates, double *source, double gamma, int dir, double *buf)
{
	double vMin;

	int ksource, kdest;

	for (ksource = 0; ksource < nStates; ksource++)
		buf[ksource] = gamma * source[ksource] - m_message[ksource];

	if (dir == m_dir) {
		for (kdest = 0; kdest < nStates; kdest++) {
			vMin = buf[0] + m_data[0 + kdest * nStates];
			for (ksource=1; ksource < nStates; ksource++) {
				if (vMin > buf[ksource] + m_data[ksource + kdest * nStates])
					vMin = buf[ksource] + m_data[ksource + kdest * nStates];
			}
			m_message[kdest] = vMin;
		}
	} else {
		for (kdest=0; kdest < nStates; kdest++) {
			vMin = buf[0] + m_data[kdest + 0 * nStates];
			for (ksource = 1; ksource < nStates; ksource++) {
				if (vMin > buf[ksource] + m_data[kdest + ksource * nStates])
					vMin = buf[ksource] + m_data[kdest + ksource * nStates];
			}
			m_message[kdest] = vMin;
		}
	}

	vMin = m_message[0];
	for (kdest = 1; kdest < nStates; kdest++)
		if (vMin > m_message[kdest]) 
			vMin = m_message[kdest];

	for (kdest=0; kdest < nStates; kdest++)
		m_message[kdest] -= vMin;


	return vMin;
}

inline void Edge::AddColumn(int nStates, int ksource, double *dest, int dir)
{
	if (dir == m_dir) {
		for (int k = 0; k < nStates; k++)
			dest[k] += m_data[ksource + k * nStates];
	} else {
		for (int k = 0; k < nStates; k++)
			dest[k] += m_data[k + ksource * nStates];
	}
}

