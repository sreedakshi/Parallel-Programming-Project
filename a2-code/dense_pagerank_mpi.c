#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <errno.h>
#include "densemat.h"
#include <mpi.h>

int main(int argc, char **argv){
  if(argc < 3){
    printf("usage: %s row_col.txt damping\n  0.0 < damping <= 1.0\n",argv[0]);
    return -1;
  }

  MPI_Init (&argc, &argv);

  int procid, total_procs;
  densemat_t *mat;
  double damping_factor = atof(argv[2]);

  MPI_Comm_size(MPI_COMM_WORLD, &total_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);


  //initializing and loading the dense matrix only onto the root proc to be scattered later
  if (procid == 0) {
    mat = densemat_load(argv[1]);

    printf("Loaded %s: %d rows, %d nonzeros\n",argv[1],mat->nrows,mat->nnz);
    printf("Beginning Computation\n\n%4s %8s %8s\n","ITER","DIFF","NORM");

  }

   //broadcasting the number of rows and cols of the matrix on the root proc so that all other procs
   //have access
   int *data = malloc(sizeof(int) * 2);
    if (procid == 0) {
      data[0] = mat->nrows;
      data[1] = mat->ncols;
    }
    MPI_Bcast(data, 2, MPI_INT, 0, MPI_COMM_WORLD);
    int rows_per_proc = data[0] / total_procs;
    if (procid != 0) {
      mat = densemat_new(data[0], data[1]); // not being used except for pointing and not using that
      // if you use if statement around scatterv, this is unnecessary

    }
    //used in the case of an uneven distribution of rows per processor
    //counts2 represents the number of rows each proc has and displs2 represents the offsets of each
    //proc's row
    int *counts2 = malloc(total_procs * sizeof(int));
    int *displs2 = malloc(total_procs * sizeof(int));
    int elements_per_proc2 = rows_per_proc;
    int surplus2 = data[0] % total_procs;
    for(int i=0; i<total_procs; i++){
      counts2[i] = (i < surplus2) ? elements_per_proc2+1 : elements_per_proc2;
      displs2[i] = (i == 0) ? 0 : displs2[i-1] + counts2[i-1];
    }

    //also used in case of an uneven distribution of rows per processor, but now considers total elements
    //by multiplying the number of cols by each processor's row number
    int *counts = malloc(total_procs * sizeof(int));
    int *displs = malloc(total_procs * sizeof(int));

    int surplus = (data[0] % total_procs);
    for(int i=0; i<total_procs; i++){
      counts[i] = (i < surplus) ? (elements_per_proc2+1)*data[1] : elements_per_proc2*data[1];
      displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }

    //if proc == 0, arg to pass in is mat->all, if not proc == 0 then pass in something else
    // THIS SURROUNDS THE SCATTERV

    //each processor allocates a densemat matrix for it to be able to receive rows from the root matrix
    densemat_t *indiv_mat = densemat_new(counts2[procid], data[1]);
    MPI_Scatterv(mat->all, counts, displs, MPI_DOUBLE, indiv_mat->all, counts[procid], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //Scatterv ok

    double *colsums = malloc(data[1] * sizeof(double));

    //each proc will initialize and determine colsums for each of its individual matrix data
    for(int c=0; c<data[1]; c++){
      colsums[c] = 0.0;
    }


    for(int r=0; r<counts2[procid]; r++){
      for(int c=0; c<data[1]; c++){
        colsums[c] += indiv_mat->data[r][c];
      }
    }


  //necessary so that all processors are updated with every other processor's column sum
  MPI_Allreduce(MPI_IN_PLACE, colsums, data[1], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  //every processor normalizing the column
  for(int r=0; r<counts2[procid]; r++){

    for(int c=0; c<data[1]; c++){
      indiv_mat->data[r][c] /= colsums[c];
      }
  }


    //every proc scales by damping factor
    double zelem = (1.0-damping_factor) / data[0];
    for(int r=0; r<counts2[procid]; r++){
        for(int c=0; c<data[1]; c++){
        if(indiv_mat->data[r][c] != 0.0){ // Scale down nonzero entries
            indiv_mat->data[r][c] *= damping_factor;
        }
        indiv_mat->data[r][c] += zelem; // Every entry increases a little
        }
    }


  //each proc needs the entire old ranks matrix and each proc contains its own cur_ranks matrix
  //with the length of how many rows each proc has
  double *old_ranks = malloc(data[0] * sizeof(double));
  double *cur_ranks_indiv = malloc(counts2[procid] * sizeof(double));
  double *cur_ranks = malloc(data[0] * sizeof(double));

  for(int c=0; c<data[1]; c++){
    cur_ranks[c] = 1.0 / data[0];
    old_ranks[c] = cur_ranks[c];
  }

  double TOL = 1e-3;
  int MAX_ITER = 10000;
  int iter;
  double change = TOL*10;
  double cur_norm;

for(iter=1; change > TOL && iter<=MAX_ITER; iter++){
    // old_ranks assigned to cur_ranks
    for(int c=0; c<data[1]; c++){
      old_ranks[c] = cur_ranks[c];
    }

    change = 0.0;
    cur_norm = 0.0;

    // Compute matrix-vector product: cur_ranks = Matrix * old_ranks

    for(int r=0; r<counts2[procid]; r++){
      // Dot product of matrix row with old_ranks column
      cur_ranks_indiv[r] = 0.0;
      for(int c=0; c<data[1]; c++){
        cur_ranks_indiv[r] += indiv_mat->data[r][c] * old_ranks[c];
      }

      double diff;

      //offset into old_ranks necessary so that each proc accesses its respective part of old_ranks
      //and not just the first part each time
      diff = cur_ranks_indiv[r] - old_ranks[r + displs2[procid]];
      change += diff>0 ? diff : -diff;
      cur_norm += cur_ranks_indiv[r]; // Tracked to detect any errors
    }
    //allgather necessary so that all procs can gather the current ranks computed on each individual processor into
    //cur_ranks to also be stored on every processor
    MPI_Allgatherv(cur_ranks_indiv, counts2[procid], MPI_DOUBLE, cur_ranks, counts2, displs2, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //allreduce necessary on change and cur_norm and necessary for all processors to have to know when to converge
    MPI_Allreduce(MPI_IN_PLACE, &cur_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (procid ==0) {
      printf("%3d: %8.2e %8.2e\n",iter,change,cur_norm);
    }
}

    if (procid == 0) {


      if(change < TOL){
      printf("CONVERGED\n");
    }
    else{
      printf("MAX ITERATION REACHED\n");
    }

    printf("\nPAGE RANKS\n");
    for(int r=0; r<data[0]; r++){
      printf("%.8f\n",cur_ranks[r]);
    }


  }

    densemat_free(mat);
    free(counts);
    free(displs);
    free(counts2);
    free(displs2);
    free(colsums);
    free(data);
    free(old_ranks);
    free(cur_ranks);
    free(cur_ranks_indiv);
    densemat_free(indiv_mat);


  MPI_Finalize();
  return 0;
}
