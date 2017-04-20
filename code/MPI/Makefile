default: sequential-pagerank.cpp
	g++ -g -std=c++11 sequential-pagerank.cpp -o spagerank
mpi: pagerank.cpp 
	mpicxx -fopenmp -std=c++11 pagerank.cpp -o pagerank
test: mpi
	mpiexec -np 4 ./pagerank 10 2 50
bigtest: mpi
	mpiexec -np 25 ./pagerank 100 2 50
clean:
	rm pagerank