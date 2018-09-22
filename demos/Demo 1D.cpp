#include "Demo 1D.h"
#include "exact.h"
#include "chain.h"
#include "tree.h"

void print_help(char *argv0)
{
	printf("Usage: %s <app>\n", argv0);
	printf("where <app> is on of the following:\n");
	printf("  exact - for exact inference and decoding\n");
	printf("  chain - for chain-structured graphical model\n");
	printf("  tree  - for tree-structured graphical model\n");
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		print_help(argv[0]);
		return 0;
	}
	std::string arg = argv[1];
	
	CDemo1D * demo = NULL;

	// Chosing demo
	if (arg == "exact") demo = new CExact();
	if (arg == "chain") demo = new CChain();
	if (arg == "tree")  demo = new CTree();
	if (!demo) {
		print_help(argv[0]);
		return 0;
	}

	demo->Main();

	// Exiting
	printf("\nPress <Enter> key to exit...");
	getchar();

	delete demo;
	return 0;
}
