#include "krongraph500.h"

#include "omp.h"
#include <mpi.h>

using namespace std;

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("%d running",rank);

	MPI_Finalize();
	return 0;
}
