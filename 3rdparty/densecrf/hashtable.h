#pragma once

#include "types.h"

class CHashTable {
public:
	CHashTable(int key_size, int nElements);
	~CHashTable(void);

	void	reset(void);
	int		find(const Mat &key, bool create = false);
	
    // Accessors
	int		size(void) const { return m_filled; }
	Mat		getKey(int i) const { return m_keys.row(i); }


private:
	void	grow(void);
	size_t	hash(const Mat &key);


private:
	int					m_capacity;
	int					m_key_size;
	int					m_filled;
	
	Mat					m_keys;
	std::vector<int>	m_vTable;	

    
private:
    // Copy semantics are disabled
    CHashTable(const CHashTable &rhs) {}
    const CHashTable & operator= (const CHashTable & rhs) { return *this; }
};
