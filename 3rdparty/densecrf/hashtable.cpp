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

namespace {
	size_t hashfunc(const Mat &key)
	{
		size_t res = 0;
		for (int i = 0; i < key.cols; i++) {
			res += key.at<short>(0, i);
			res *= 1664525;
		}
		return res;
	}

	size_t index(const Mat &key, size_t array_size)
	{
		size_t hash = hashfunc(key);
		size_t index = hash % array_size;
		return index;
	}
}

int CHashTable::find(const Mat &key) const
{
	// Get the hash value
	size_t idx = index(key(Rect(0, 0, m_key_size, 1)), m_capacity);
	
	// Find the element with he right key, using linear probing
	while (true) {
		int value = m_vTable[idx];
		if (value == -1) return value;
		
		// Check if the current key is The One
		bool good = true;
		
		const short *pKeys = m_keys.ptr<short>(value);
		const short *pKey = key.ptr<short>(0);
		for (size_t i = 0; i < m_key_size && good; i++)
			if (pKeys[i] != pKey[i])
				good = false;
		
		if (good) return value;
		
		// Continue searching
		idx++;
		if (idx == m_capacity) idx = 0;
	}
}

void CHashTable::insert(const Mat &key, int value)
{
	if (2 * m_filled >= m_capacity) grow();

	// Get the hash value
	size_t idx = index(key(Rect(0, 0, m_key_size, 1)), m_capacity);

	while (true) {
		int value = m_vTable[idx];
		if (value == -1) {
			// Insert a new key and return the new id
			key(Rect(0, 0, m_key_size, 1)).copyTo(m_keys.row(m_filled));

			m_vTable[idx] = static_cast<int>(m_filled);
			m_filled++;
			return;
		}
		// Continue searching empty cell
		idx++;
		if (idx == m_capacity) idx = 0;
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
		int value = old_table[i];
		if (value >= 0) {
			size_t idx = index(getKey(value), m_capacity);
			for (; m_vTable[idx] >= 0; idx = idx < m_capacity - 1 ? idx + 1 : 0);
			m_vTable[idx] = value;
		}
	} // i
}
