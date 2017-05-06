#include "csc.h"
#include "krongraph500.h"

#include "omp.h"
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

void prefix_sum(int a[], int s[], int n) {
    int *sums;
    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        #pragma omp single
        {
            sums = new int[nthreads+1];
            sums[0] = 0;
        }
        int sum = 0;
        #pragma omp for schedule(static) nowait 
        for(int i=0; i<n; i++) {
            sum += a[i];
            s[i] = sum;
        }
        sums[ithread+1] = sum;
        #pragma omp barrier
        int offset = 0;
        for(int i=0; i<(ithread+1); i++) {
            offset += sums[i];
        }

        #pragma omp for schedule(static) 
        for(int i=0; i<n; i++) {
            s[i] += offset;
        }
    }
    delete[] sums;
}

csc_matrix<long>* getRandomGraph(int scale, int edgesPerNode, MPI_Comm comm){
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	long n = (1ul << scale)/size;
	vector<tuple<long,long>> edges = kronecker<long>(scale, edgesPerNode, size);

	sort(edges.begin(), edges.end()); 

	vector<tuple<long, long>>::iterator current_edge = edges.begin();
	int* sendcounts = new int[size];
	for(int i = 0; i < size; i++){
		sendcounts[i] = 0;
		while(current_edge != edges.end() && get<0>(*current_edge) < (i+1)*n){
			sendcounts[i] += 2;
			++current_edge;
		}
	}

	int* recvcounts = new int[size];
	MPI_Alltoall(sendcounts, 1, MPI_INT,recvcounts,1,MPI_INT,comm);
	//convert to omp reduction
	int sendtotal = 0;
	int recvtotal = 0;
	for(int i = 0; i < size; i++){
		recvtotal += recvcounts[i];
		sendtotal += sendcounts[i];
	}

	long* data = new long[recvtotal];
	long* sendbuffer = new long[sendtotal];
	int pos = 0;
	vector<tuple<long,long>>::iterator it = edges.begin();
	while(it != edges.end()){
		sendbuffer[pos] = get<0>(*it);
		sendbuffer[pos+1] = get<1>(*it);
		pos += 2;
		++it;
	}
	
	int* recvdispl = new int[size];
	int* senddispl = new int[size];
	prefix_sum(recvcounts,recvdispl,size);
	prefix_sum(sendcounts,senddispl,size);
	MPI_Alltoallv(sendbuffer,sendcounts,senddispl, MPI_INT,data,recvcounts,recvdispl, MPI_INT, comm);

	vector<tuple<long,long>> local_edges;
	local_edges.reserve(recvtotal/2);
	for(int i = 0; i < recvtotal; i += 2){
		local_edges.push_back(tuple<long,long>(data[i],data[i+1]));
	}
	// TODO make sure to clean up buffers 


	vector<tuple<long,long, double>> matrix;
  	matrix.reserve(edges.size());
    auto &prev = edges[0];
    long count = 1;

    for (size_t i = 1; i < edges.size(); i++) {
      if (prev == edges[i]) {
        count++;
      } else {
        // Transpose
        matrix.push_back(tuple<long,long,long>(get<0>(prev), get<1>(prev), count));
        prev = edges[i];
        count = 1;
      }
    }
    matrix.push_back(tuple<long,long,long>(get<0>(prev), get<1>(prev), count));
  	sort(matrix.begin(), matrix.end());
	csc_matrix<long>* result = new csc_matrix<long>(n, matrix);
	return result;
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

	int scale = 4;
	int edgesPerNode = 2;
	int iterations = 50;
	double teleport_probability = 0.85;
	if(argc > 3){
		scale = atoi(argv[1]);
		edgesPerNode = atoi(argv[2]);
		iterations = atoi(argv[3]);
	}
	
	MPI_Comm comm = MPI_COMM_WORLD;
	int wrank,wsize;
	MPI_Comm_rank(comm, &wrank);
	MPI_Comm_size(comm, &wsize);
	if(argc > 4 && wrank ==0){
		printf("Too many args! Usage: pagerank <scale> <edgesPerNode> <iterations>");
	}



	unsigned long n = (1ul << scale)/wsize;

	csc_matrix<long>* matrix = getRandomGraph(scale,edgesPerNode,comm);
	
	/*
	normalize_matrix(n,&matrix,col_comm);

	MPI_Barrier(comm);
	double start = MPI_Wtime();
	double* r = compute_pagerank<long>(&matrix, n,teleport_probability,iterations, col_comm);
	MPI_Barrier(comm);
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
	*/
	MPI_Finalize();
	return 0;
}
