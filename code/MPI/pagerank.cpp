#include "csc.h"

//#include "omp.h"
//#include <mpi.h>
#include <cstddef>
#include <climits>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <tuple>
#include <iostream>

using namespace std;

template<class T>
T getRandomInt(T max){
	static thread_local mt19937 generator;
    uniform_int_distribution<T> distribution(0,max);
    return distribution(generator);
}

template<class T>
tuple<T,T,int> getRandomEdge(T start, T n){
	T dest = getRandomInt<T>(n-1);
	int inDegree = getRandomInt<int>(100);
	//printf("created random edge from %d to %d with value %d\n", start, dest, inDegree);
    return tuple<T,T,int>(start, dest, inDegree);
}

double getRandomReal(){
	static thread_local mt19937 generator;
	uniform_real_distribution<double> distribution(0,1);
	return distribution(generator);
}

csc_matrix<long> getRandomGraph(int n, int edgesPerNode,int startCol, int endCol, int colSize){
	vector<tuple<long,long,int>> edges;

	for(int i = startCol; i < endCol; i++){
		for(int j = 0; j < edgesPerNode; j++){
			bool shouldGenerate = getRandomReal() < 1.0/colSize;
			if(shouldGenerate){
				edges.push_back(getRandomEdge<long>(i,n));
			}
		}
	}

	sort(edges.begin(), edges.end()); //column based so sorting works, small list so parallelism doesn't help

	/*for(vector<tuple<long, long, int>>::const_iterator it = edges.begin(); it!=edges.end();++it){
		tuple<long,long,int> edge = *it;
	tuple<long,long,int>* prev = nullptr;
		printf("edge from %d to %d with value %d\n",get<0>(edge),get<1>(edge),get<2>(edge));
	}*/

	vector<tuple<long, long, int>>::iterator it = edges.begin();
	tuple<long,long,int> prev = *it;
	++it;
	for(; it != edges.end(); ){
		tuple<long, long, int> edge = *it;
		int prevCol,prevRow, col,row;
		prevCol = get<0>(prev);
		prevRow = get<1>(prev);
		col = get<0>(edge);
		row = get<1>(edge);
		if (prevCol == col && prevRow == row) {
			it = edges.erase(it);
		}else{
			prev = edge;
			++it;
		}

	}
	return csc_matrix<long>(n,edges);
}

void normalize_matrix(int n, csc_matrix<long>* matrix){
	for(int i = 0; i < n; i++){
		double sum = 0.0;
		for(int j = matrix->col_starts[i]; j < matrix->col_starts[i+1];j++){
			sum += matrix->vals[j];
		}
		for(int j = matrix->col_starts[i]; j < matrix->col_starts[i+1]; j++){
			matrix->vals[j] = (double)matrix->vals[j] / (double)sum;
		}
	}
}

template<class T>
void printVector(vector<T> vec){
	for (typename vector<T>::const_iterator i = vec.begin(); i != vec.end(); ++i)
    cout << *i << ' ';
	cout << endl;
}

int main(int argc, char** argv){
	//MPI_Init(&argc, &argv);

	int n = 4;
	int edgesPerNode = 2;
	if(argc > 2){
		n = atoi(argv[1]);
		edgesPerNode = atoi(argv[2]);
	}
	
	int rank = 0;
	//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(argc > 3 && rank ==0){
		printf("Too many args! Usage: pagerank n edgesPerNode");
	}

	csc_matrix<long> matrix = getRandomGraph(n,edgesPerNode,0,n,1);

	printVector(matrix.col_starts);
	printVector(matrix.rows);
	printVector(matrix.vals);

	printf("normalizing matrix...\n");
	normalize_matrix(n,&matrix);

	printVector(matrix.col_starts);
	printVector(matrix.rows);
	printVector(matrix.vals);


	//MPI_Finalize();
	return 0;
}
