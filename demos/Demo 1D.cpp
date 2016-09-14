#include "Demo 1D.h"
#include "exact.h"
#include "chain.h"
#include "tree.h"

void print_help(void)
{
	printf("Usage: \"Demo 1D.exe\" <app>\n");
	printf("where <app> is on of the following:\n");
	printf("  exact - for \n");
	printf("  chain - for \n");
	printf("  tree  - for \n");
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		print_help();
		return 0;
	}
	std::string arg = argv[1];
	
	CDemo1D * demo = NULL;

	// Chosing demo
	if (arg == "exact") demo = new CExact();
	if (arg == "chain") demo = new CChain();
	if (arg == "tree")  demo = new CTree();
	if (!demo) {
		print_help();
		return 0;
	}

	demo->Main();

	// Exiting
	printf("\nPress <Enter> key to exit...");
	getchar();

	delete demo;
	return 0;
}