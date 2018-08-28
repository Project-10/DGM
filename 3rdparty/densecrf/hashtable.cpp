#include "hashtable.h"

// Constructor
CHashTable::CHashTable(int key_size, int nElements) : m_key_size(key_size), m_capacity(nElements / 24), m_filled(0)
{
	m_vTable = std::vector<int>(m_capacity, -1);
	m_keys   = Mat(m_capacity / 2 + 10, m_key_size, CV_16SC1);
}

// Destructor
CHashTable::~CHashTable(void)
{}

void CHashTable::reset(void)
{
	m_filled = 0;
	std::fill(m_vTable.begin(), m_vTable.end(), -1);
}

// if add: returns list size - 1
int CHashTable::find(const Mat &key, bool create)
{
	if (2 * m_filled >= m_capacity) grow();
	
	// Get the hash value
	size_t h = hash(key) % m_capacity;
	
	// Find the element with he right key, using linear probing
	while (true) {
		int e = m_vTable[h];
		if (e == -1) {
			if (create) {
				// Insert a new key and return the new id
				short *pKeys = m_keys.ptr<short>(m_filled);
				const short *pKey = key.ptr<short>(0);
				for (size_t i = 0; i < m_key_size; i++)
					pKeys[i] = pKey[i];			// TODO
				
				m_vTable[h] = static_cast<int>(m_filled);
				m_filled++;
				return m_vTable[h];
			}
			else
				return -1;
		}
		
		// Check if the current key is The One
		bool good = true;
		
		short *pKeys = m_keys.ptr<short>(e);
		const short *pKey = key.ptr<short>(0);
		for (size_t i = 0; i < m_key_size && good; i++)
			if (pKeys[i] != pKey[i])
				good = false;
		
		if (good) return e;
		
		// Continue searching
		h++;
		if (h == m_capacity) h = 0;
	}
}

void CHashTable::grow(void)
{
	// Swap out the old memory
	std::vector<int>	old_table = m_vTable;
	
	size_t	old_capacity = m_capacity;
	m_capacity *= 2;
	
	// Allocate the new memory
	copyMakeBorder(m_keys, m_keys, 0, (old_capacity + 10) * m_key_size - m_keys.rows, 0, 0, BORDER_CONSTANT);
	m_vTable = std::vector<int>(m_capacity, -1);

	// Reinsert each element
	for (int i = 0; i < old_capacity; i++) {
		int e = old_table[i];
		if (e >= 0) {
			size_t h = hash(getKey(e)) % m_capacity;
			for (; m_vTable[h] >= 0; h = h < m_capacity - 1 ? h + 1 : 0);
			m_vTable[h] = e;
		}
	} // i
}

size_t CHashTable::hash(const Mat &key)
{
	size_t res = 0;
	for (int i = 0; i < m_key_size; i++) {
		res += key.at<short>(0, i);		
		res *= 1664525;
	}
	return res;
}
