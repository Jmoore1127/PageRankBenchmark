#include "csc.h"

//#include "omp.h"
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
	static thread_local random_device rd;
	static thread_local mt19937 generator(rd());
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
	static thread_local random_device rd;
	static thread_local mt19937 generator(rd());
	uniform_real_distribution<double> distribution(0,1);
	return distribution(generator);
}

csc_matrix<long> getRandomGraph(int n, int edgesPerNode, int cols, int colSize){
	vector<tuple<long,long,int>> edges;

	for(int i = 0; i < cols; i++){
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
	double col_sums[n];
	for(int i = 0; i < n; i++){
		double sum = 0.0;
		for(int j = matrix->col_starts[i]; j < matrix->col_starts[i+1];j++){
			sum += matrix->vals[j];
		}
		col_sums[i] = sum;
		
	}
	for(int i = 0; i < n; i++){
		if(col_sums[i] != 0){
			for(int j = matrix->col_starts[i]; j < matrix->col_starts[i+1]; j++){
				matrix->vals[j] = (double)matrix->vals[j] / (double)col_sums[i];
			}
		}
	}
}

template<class T>
void printMatrix(csc_matrix<T> mat,bool abbreviated){
	int m = mat.col_starts.size();
	int entries = mat.vals.size();
	string format = "==========\nMatrix:\n";
	format += "Vals: [";
	for(int i = 0; i < entries; i++ ){
		format += to_string(mat.vals.at(i))+", ";
	}
	format += "]\n";

	if(!abbreviated){
		format += "Rows: [";
		for(int i = 0; i < entries; i++ ){
			format += to_string(mat.rows.at(i))+", ";
		}
		format += "]\n";

		format += "ColStarts: [";
		for(int i = 0; i < m; i++){
			format += to_string(mat.col_starts.at(i))+", ";
		}
		format += "]\n";
	}
	format += "==========\n";
	cout << format;
}

template<class T>
double* compute_pagerank(csc_matrix<T>* transition_matrix, int n, double beta, int iterations){
	double* r = new double[n];
	for(int i = 0; i < n; i++){
		r[i] = getRandomReal();
	}

	for(int i = 0; i < iterations; i++){
		double* vals = new double[n];
		for(int j = 0; j < n; j++){
			double col_sum = 0.0;
			for(int k = transition_matrix->col_starts.at(j); k < transition_matrix->col_starts.at(j+1); k++){
				col_sum += r[transition_matrix->rows.at(k)] * transition_matrix->vals.at(k);
			}
			vals[j] = col_sum * beta + (1.0 - beta)/ n;
		}
		double* temp = r;
		r = vals;
		delete[] temp;
	}
	return r;
}

int main(int argc, char** argv){

	int N = 4;
	int edgesPerNode = 2;
	int iterations = 50;
	double teleport_probability = 0.85;
	if(argc > 3){
		N = atoi(argv[1]);
		edgesPerNode = atoi(argv[2]);
		iterations = atoi(argv[3]);
	}
	
	if(argc > 4){
		printf("Too many args! Usage: pagerank n edgesPerNode");
	}


	int n = N;

	csc_matrix<long> matrix = getRandomGraph(N,edgesPerNode,n,1);
	printMatrix(matrix,false);
	
	normalize_matrix(n,&matrix);
	printMatrix(matrix,false);

	double* r = compute_pagerank<long>(&matrix, n,teleport_probability,iterations);
	
	string output = "r = [";
	for(int i = 0; i < N; i++){
		output += to_string(r[i]) + ", ";
	}
	output += "]\n";
	cout << output;
	return 0;
}
