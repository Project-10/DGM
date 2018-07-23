#pragma once

#include "types.h"

class HashTable {
public:
	HashTable(int key_size, int n_elements);
	~HashTable(void);

	size_t		  size(void) const { return m_filled; }
	void		  reset(void);
	int			  find(const short * k, bool create = false);
	const short	* getKey(int i) const { return m_pKeys + i * m_key_size; }


protected:
	size_t	  m_key_size;
	size_t	  m_capacity;
	size_t	  m_filled;
	short	* m_pKeys;
	int		* m_pTable;

	void	grow(void);
	size_t	hash(const short * k);


private:
	HashTable(const HashTable & o);

};