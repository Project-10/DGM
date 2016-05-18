#pragma once

#include <string.h>
#include <assert.h>

class TypeGeneral
{
private:
	struct Vector;		// node parameters and messages
	struct Edge;		// stores edge information and either forward or backward message

	//////////////////////////////////////////////////////////////////////////////////
	////////////////////////// Visible only to MRFEnergy /////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////

private:
	friend class MRFEnergy;

	struct Vector
	{
		static int	GetSizeInBytes(int K);				// returns -1 if invalid K's
		void		Initialize(int K, double * data);	// called once when user adds a node

		void		SetZero(int K);                     // set this[k] = 0
		void		Copy(int K, Vector* V);             // set this[k] = V[k]
		void		Add(int K, Vector* V);              // set this[k] = this[k] + V[k]
		double		GetValue(int K, int k);             // return this[k]
		double		ComputeMin(int K, int& kMin);       // return min_k { this[k] }, set kMin
		double		ComputeAndSubtractMin(int K);       // same as previous, but additionally set this[k] -= vMin (and kMin is not returned)

		double		GetArrayValue(int K, int k);		// note: k is an integer in [0..GetArraySize()-1].
																					// For Potts functions GetArrayValue() and GetValue() are the same,
																					// but they are different for, say, 2-dimensional labels.
		void		SetArrayValue(int K, int k, double x);

	private:
	friend struct Edge;
		double		m_data[1];														// actual size is MRFEnergy::m_Kglobal
	};

	struct Edge
	{
		static int	GetSizeInBytes(int Ki, int Kj, double * data);	// returns -1 if invalid data
		static int	GetBufSizeInBytes(int vectorMaxSizeInBytes);									// returns size of buffer need for UpdateMessage()
		void		Initialize(int Ki, int Kj, double * data, Vector* Di, Vector* Dj); // called once when user adds an edge
		Vector	  * GetMessagePtr();
		void		Swap(int Ki, int Kj);							// if the client calls this function, then the meaning of 'dir'
																									// in distance transform functions is swapped

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
		// If Edge::Swap has been called odd number of times, then the meaning of dir is swapped.
		//
		// Vector 'source' must not be modified. Function may use 'buf' as a temporary storage.
		double UpdateMessage(int Ksource, int Kdest, Vector* source, double gamma, int dir, void* buf);

		// If dir==0, then sets dest[kj] += V(ksource,kj).
		// If dir==1, then sets dest[ki] += V(ki,ksource).
		// If Swap() has been called odd number of times, then the meaning of dir is swapped.
		void AddColumn(int Ksource, int Kdest, int ksource, Vector* dest, int dir);

	protected:
		Vector	* m_message;

	private:
		int		m_dir;				// 0 if Swap() was called even number of times, 1 otherwise
		double	m_data[1];			// array of size Ki * Kj
	};
};

//////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Implementation ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

///////////////////// Vector ///////////////////////

inline int TypeGeneral::Vector::GetSizeInBytes(int K)
{
	return (K < 1) ? -1 : K * sizeof(double);
}

inline void TypeGeneral::Vector::Initialize(int K, double * data)
{
	memcpy(m_data, data, K * sizeof(double));
}

inline void TypeGeneral::Vector::SetZero(int K)
{
	memset(m_data, 0, K * sizeof(double));
}

inline void TypeGeneral::Vector::Copy(int K, Vector * V)
{
	memcpy(m_data, V->m_data, K * sizeof(double));
}

inline void TypeGeneral::Vector::Add(int K, Vector * V)
{
	for (int k = 0; k < K; k++)
		m_data[k] += V->m_data[k];
}

inline double TypeGeneral::Vector::GetValue(int K, int k)
{
	assert(k>=0 && k<K.m_K);
	return m_data[k];
}

inline double TypeGeneral::Vector::ComputeMin(int K, int& kMin)
{
	double vMin = m_data[0];
	kMin = 0;
	for (int k = 1; k < K; k++)
	{
		if (vMin > m_data[k])
		{
			vMin = m_data[k];
			kMin = k;
		}
	}

	return vMin;
}

inline double TypeGeneral::Vector::ComputeAndSubtractMin(int K)
{
	double vMin = m_data[0];

	for (int k = 1; k < K; k++)
		if (vMin > m_data[k]) vMin = m_data[k];

	for (int k = 0; k < K; k++)
		m_data[k] -= vMin;

	return vMin;
}

inline double TypeGeneral::Vector::GetArrayValue(int K, int k)
{
	assert(k >= 0 && k < K);
	return m_data[k];
}

inline void TypeGeneral::Vector::SetArrayValue(int K, int k, double x)
{
	assert(k>=0 && k<K.m_K);
	m_data[k] = x;
}

///////////////////// EdgeDataAndMessage implementation /////////////////////////

inline int TypeGeneral::Edge::GetSizeInBytes(int Ki, int Kj, double * data)
{
	int messageSizeInBytes = ((Ki > Kj) ? Ki : Kj) * sizeof(double);
	return sizeof(Edge) - sizeof(double) + Ki * Kj * sizeof(double) + messageSizeInBytes;
}

inline int TypeGeneral::Edge::GetBufSizeInBytes(int vectorMaxSizeInBytes)
{
	return vectorMaxSizeInBytes;
}

inline void TypeGeneral::Edge::Initialize(int Ki, int Kj, double * data, Vector* Di, Vector* Dj)
{
	m_dir = 0;
	memcpy(this->m_data, data, Ki * Kj * sizeof(double));
	m_message = (Vector*)((char*)this + sizeof(Edge) - sizeof(double) + Ki * Kj * sizeof(double));

	memset(m_message->m_data, 0, ((Ki > Kj) ? Ki : Kj)*sizeof(double));
}

inline TypeGeneral::Vector * TypeGeneral::Edge::GetMessagePtr()
{
	return m_message;
}

inline void TypeGeneral::Edge::Swap(int Ki, int Kj)
{
	m_dir = 1 - m_dir;
}

inline double TypeGeneral::Edge::UpdateMessage(int Ksource, int Kdest, Vector * source, double gamma, int dir, void * _buf)
{
	Vector * buf = (Vector *) _buf;
	double vMin;

	int ksource, kdest;

	for (ksource = 0; ksource < Ksource; ksource++)
	{
		buf->m_data[ksource] = gamma*source->m_data[ksource] - m_message->m_data[ksource];
	}

	if (dir == this->m_dir)
	{
		for (kdest = 0; kdest < Kdest; kdest++)
		{
			vMin = buf->m_data[0] + m_data[0 + kdest * Ksource];
			for (ksource=1; ksource < Ksource; ksource++)
			{
				if (vMin > buf->m_data[ksource] + m_data[ksource + kdest * Ksource])
				{
					vMin = buf->m_data[ksource] + m_data[ksource + kdest * Ksource];
				}
			}
			m_message->m_data[kdest] = vMin;
		}
	}
	else
	{
		for (kdest=0; kdest < Kdest; kdest++)
		{
			vMin = buf->m_data[0] + m_data[kdest + 0 * Kdest];
			for (ksource=1; ksource < Ksource; ksource++)
			{
				if (vMin > buf->m_data[ksource] + m_data[kdest + ksource * Kdest])
				{
					vMin = buf->m_data[ksource] + m_data[kdest + ksource * Kdest];
				}
			}
			m_message->m_data[kdest] = vMin;
		}
	}

	vMin = m_message->m_data[0];
	for (kdest = 1; kdest < Kdest; kdest++)
	{
		if (vMin > m_message->m_data[kdest])
		{
			vMin = m_message->m_data[kdest];
		}
	}

	for (kdest=0; kdest < Kdest; kdest++)
		m_message->m_data[kdest] -= vMin;


	return vMin;
}

inline void TypeGeneral::Edge::AddColumn(int Ksource, int Kdest, int ksource, Vector* dest, int dir)
{
	assert(ksource >= 0 && ksource < Ksource);

	if (dir == this->m_dir) {
		for (int k = 0; k < Kdest; k++)
			dest->m_data[k] += m_data[ksource + k * Ksource];
	} else {
		for (int k = 0; k < Kdest; k++)
			dest->m_data[k] += m_data[k + ksource * Kdest];
	}
}

