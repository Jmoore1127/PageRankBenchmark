#include "csc.h"

//#include "omp.h"
#include <mpi.h>
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
    return tuple<T,T,int>(start, dest, 1.0);
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
			get<2>(prev) = get<2>(prev) + 1;
			it = edges.erase(it);
		}else{
			prev = edge;
			++it;
		}

	}
	return csc_matrix<long>(n,edges);
}

void normalize_matrix(int n, csc_matrix<long>* matrix, MPI_Comm comm){
	double col_sums[n];
	for(int i = 0; i < n; i++){
		double sum = 0.0;
		for(int j = matrix->col_starts[i]; j < matrix->col_starts[i+1];j++){
			sum += matrix->vals[j];
		}
		col_sums[i] = sum;
		
	}
	double total_sums[n];
	MPI_Reduce(&col_sums,&total_sums, n, MPI_DOUBLE, MPI_SUM, 0, comm);
	MPI_Bcast(&total_sums,n,MPI_DOUBLE, 0, comm);
	for(int i = 0; i < n; i++){
		if(total_sums[i]!= 0){
			for(int j = matrix->col_starts[i]; j < matrix->col_starts[i+1]; j++){
					matrix->vals[j] = (double)matrix->vals[j] / (double)total_sums[i];
			}
		}
	}
}

template<class T>
void printMatrix(csc_matrix<T> mat,int rank,bool abbreviated){
	int m = mat.col_starts.size();
	int entries = mat.vals.size();
	string format = "==========\nRank "+to_string(rank)+ " Matrix:\n";
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
double* compute_pagerank(csc_matrix<T>* transition_matrix, int n, double beta, int iterations, MPI_Comm comm){
	int rank;
	MPI_Comm_rank(comm,&rank);
	double* r = new double[n];
	if(rank == 0){
		for(int i = 0; i < n; i++){
			r[i] = getRandomReal();
		}
	}
	
	MPI_Bcast(r,n,MPI_DOUBLE, 0,comm);

	for(int i = 0; i < iterations; i++){
		double* vals = new double[n];
		for(int j = 0; j < n; j++){
			double col_sum = 0.0;
			for(int k = transition_matrix->col_starts.at(j); k < transition_matrix->col_starts.at(j+1); k++){
				col_sum += r[transition_matrix->rows.at(k)] * transition_matrix->vals.at(k);
			}
			vals[j] = col_sum;//* beta + (1.0 - beta)/ n;
		}
		MPI_Reduce(vals,r, n, MPI_DOUBLE, MPI_SUM, 0, comm);
		//delete[] vals;
		if(rank == 0){
			for(int i = 0; i < n; i++){
				r[n] = r[n] * beta + (1.0 - beta)/n;
			}
		}
		MPI_Bcast(r,n,MPI_DOUBLE, 0, comm);		
	}
	
	return r;
}

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);

	int N = 4;
	int edgesPerNode = 2;
	int iterations = 50;
	double teleport_probability = 0.85;
	if(argc > 3){
		N = atoi(argv[1]);
		edgesPerNode = atoi(argv[2]);
		iterations = atoi(argv[3]);
	}
	
	int wrank,wsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
	MPI_Comm_size(MPI_COMM_WORLD, &wsize);
	if(argc > 4 && wrank ==0){
		printf("Too many args! Usage: pagerank n edgesPerNode");
	}

	int col_size = sqrt(wsize);
	if(round(col_size) * round(col_size) != wsize && wrank == 0){
		printf("Error: must use a perfect square number of tasks.");
		return 1;
	}

	MPI_Comm row_comm, col_comm;
	MPI_Comm_split(MPI_COMM_WORLD,wrank/col_size,wrank,&row_comm);
	MPI_Comm_split(MPI_COMM_WORLD,wrank%col_size,wrank,&col_comm);

	int n = N/col_size;

	csc_matrix<long> matrix = getRandomGraph(N,edgesPerNode,n,col_size);
	
	normalize_matrix(n,&matrix,col_comm);

	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	double* r = compute_pagerank<long>(&matrix, n,teleport_probability,iterations, col_comm);
	MPI_Barrier(MPI_COMM_WORLD);
	double time = MPI_Wtime() - start;

	if(wrank == 0){
		cout << "Finished " << N*n*edgesPerNode << " edges with " << iterations << " iterations in " << time << " seconds" << endl;
	}

	if(wrank < col_size){
		double* R = new double[N];
		MPI_Gather(r, n, MPI_DOUBLE, R, n,MPI_DOUBLE, 0, row_comm);
		if(wrank == 0){
			string output = "r = [";
			for(int i = 0; i < N; i++){
				output += to_string(R[i]) + ", ";
			}
			output += "]\n";
			cout << output;
		}
	}
	
	MPI_Finalize();
	return 0;
}
