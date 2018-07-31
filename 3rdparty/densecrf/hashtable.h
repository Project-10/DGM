#pragma once

#include "types.h"

class CHashTable {
public:
	CHashTable(int key_size, size_t nElements);
	~CHashTable(void);

	void		  reset(void);
	int			  find(const short *key, bool create = false);
	
    // Accessors
	size_t		  size(void) const{ return m_filled; }
	const short	* getKey(int i) const { return m_pKeys + i * m_key_size; }


private:
	void	grow(void);
	size_t	hash(const short *key);


private:
	size_t	  m_key_size;
	size_t	  m_capacity;
	size_t	  m_filled;
	short	* m_pKeys;
	int		* m_pTable;

    
private:
    // Copy semantics are disabled
    CHashTable(const CHashTable &rhs) {}
    const CHashTable & operator= (const CHashTable & rhs) { return *this; }
};
