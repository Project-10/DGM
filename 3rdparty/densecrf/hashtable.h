#pragma once

#include "types.h"

class CHashTable {
public:
	CHashTable(int key_size, size_t nElements);
	~CHashTable(void);

	void		  reset(void);
	int			  find(const std::vector<short> &key, bool create = false);
	
    // Accessors
	size_t		  size(void) const{ return m_filled; }
	const short	* getKey(int i) const { return m_pKeys + i * m_key_size; }


private:
	void	grow(void);
	size_t	hash(const std::vector<short> &key);


private:
	size_t	  m_key_size;
	size_t	  m_capacity;
	size_t	  m_filled;
	short	* m_pKeys;			// TODO: sub
	int		* m_pTable;			// TODO: sub

    
private:
    // Copy semantics are disabled
    CHashTable(const CHashTable &rhs) {}
    const CHashTable & operator= (const CHashTable & rhs) { return *this; }
};
