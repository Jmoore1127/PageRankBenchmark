default: pagerank.cpp
	g++ -g -std=c++11 pagerank.cpp -o pagerank
mpi: pagerank.cpp 
	mpicxx -fopenmp -std=c++11 pagerank.cpp -o pagerank
clean:
	rm pagerank