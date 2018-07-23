#pragma once

#include "types.h"

class CHashTable {
public:
	CHashTable(int key_size, int n_elements);
	~CHashTable(void);

	size_t		  size(void) const { return m_filled; }
	void		  reset(void);
	int			  find(const short * k, bool create = false);
	const short	* getKey(int i) const { return m_pKeys + i * m_key_size; }


protected:
	void	grow(void);
	size_t	hash(const short * k);


protected:
	size_t	  m_key_size;
	size_t	  m_capacity;
	size_t	  m_filled;
	short	* m_pKeys;
	int		* m_pTable;


private:
	CHashTable(const CHashTable & o);
};
