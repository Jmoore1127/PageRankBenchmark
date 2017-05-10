default: sequential-pagerank.cpp
	g++ -g -std=c++11 sequential-pagerank.cpp krongraph500.cpp -o spagerank
mpi: pagerank.cpp krongraph500.cpp
	mpicxx -fopenmp -std=c++11 pagerank.cpp krongraph500.cpp -o pagerank
mock: pagerank.cpp krongraph500mock.cpp
	mpicxx -fopenmp -std=c++11 pagerank.cpp krongraph500mock.cpp -o pagerank
smalltest: mpi
	mpiexec -np 1 ./pagerank 5 2 50
test: mpi
	mpiexec -np 4 ./pagerank 10 2 50
bigtest: mpi
	mpiexec -np 25 ./pagerank 100 2 50
clean:
	rm pagerank