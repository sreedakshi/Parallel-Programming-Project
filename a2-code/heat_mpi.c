#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv){
  if(argc < 4){
    printf("usage: %s max_time width print\n", argv[0]);
    printf("  max_time: int\n");
    printf("  width: int\n");
    printf("  print: 1 print output, 0 no printing\n");
    return 0;
  }

  MPI_Init (&argc, &argv);

  int procid, total_procs;

  MPI_Comm_size(MPI_COMM_WORLD, &total_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

 //printf("%d: procid", procid); sdk

  int max_time = atoi(argv[1]); // Number of time steps to simulate
  int width = atoi(argv[2]);    // Number of cells in the rod
  int print = atoi(argv[3]);
  double initial_temp = 50.0;   // Initial temp of internal cells
  double L_bound_temp = 20.0;   // Constant temp at Left end of rod
  double R_bound_temp = 10.0;   // Constant temp at Right end of rod
  double k = 0.5;               // thermal conductivity constant
  double **H;                   // 2D array of temps at times/locations
  double **rbuf;

  int total_columns = width;
  int elements_per_proc = total_columns / total_procs;

   // Allocate memory
  H = malloc(sizeof(double*)*max_time);
  int t,p;
  for(t=0;t<max_time;t++){
    H[t] = malloc(sizeof(double*)*elements_per_proc); //each processor allocs enough of matrix H to hold its
                                                      //respective elements
  }

//set everything to be initial temp
for (int i = 0; i < elements_per_proc; i++) {
   H[0][i] = initial_temp; //the first row has the initial temp in every element
}

if (procid == 0) {
  for(t=0; t<max_time; t++){
    H[t][0] = L_bound_temp; // in every row, the first element is set to 20.0
   }
}

if (procid == total_procs - 1) {
  for(t=0; t<max_time; t++){
    H[t][elements_per_proc-1] = R_bound_temp; // in every row, the right most element is set to 10.0
   }
}


 int right_partner = procid + 1;
 int left_partner = procid - 1;
   // Simulate the temperature changes for internal cells
  for(t=0; t<max_time-1; t++){
    for(p=1; p<elements_per_proc-1; p++){
      double left_diff  = H[t][p] - H[t][p-1];
      double right_diff = H[t][p] - H[t][p+1];
      double delta = -k*( left_diff + right_diff );
      H[t+1][p] = H[t][p] + delta; //computation of inner elements in each row - code previously given
    }
    double left_receive;
    double right_receive; 
    if (total_procs > 1) {
    	if (procid == 0) { //leftmost processor

        	double SEND_INFO = H[t][elements_per_proc-1]; // the right most element in the left proc 
        	right_receive = H[t+1][elements_per_proc-1]; // value of position where the info will be received from second processor
        	MPI_Sendrecv(&SEND_INFO, 1, MPI_DOUBLE, right_partner, 1, &right_receive, 1, MPI_DOUBLE, right_partner, 1, 
        	MPI_COMM_WORLD, MPI_STATUS_IGNORE); // this call handles sending the info in send_info into the second proc's respective position
          // and receives into the right_receive position from what is being sent from the left of second proc

          double left1 = H[t][elements_per_proc-1] - H[t][elements_per_proc-2]; 
          double right1 = H[t][elements_per_proc-1] - right_receive; 
          double delta1 =  -k*( left1 + right1);
          H[t+1][elements_per_proc-1] = H[t][elements_per_proc-1] + delta1; //this is where the computation to determine the right most element of
                                                                            // left processor happens

    	}
    	else if ((procid > 0 && procid < total_procs - 1) && (procid % 2 == 0)) { //middle processors and check for if even to send left first
    		  double SEND_INFO1 = H[t][0]; // storing left element
        	left_receive = H[t+1][0]; // value of position where receiving will happen on left hand side of middle proc
        	MPI_Sendrecv(&SEND_INFO1, 1, MPI_DOUBLE, left_partner, 1, &left_receive, 1, MPI_DOUBLE, left_partner, 1,
        	MPI_COMM_WORLD, MPI_STATUS_IGNORE); // this sendrecv call handles sending the top left element and 
                                              // receives into the bottom left position from left partner proc

          double left =  H[t][0] - left_receive;
          double right = H[t][0] - H[t][1];
          double delta = -k*( left + right );
          H[t+1][0] = H[t][0] + delta; //computation to determine the value of the left boundary elements of middle proc
          	
        	double SEND_INFO = H[t][elements_per_proc-1]; // value to send to right partner proc
        	right_receive = H[t+1][elements_per_proc-1]; // value of position where receiving will happend when right partner proc sends
        	MPI_Sendrecv(&SEND_INFO, 1, MPI_DOUBLE, right_partner, 1, &right_receive, 1, MPI_DOUBLE, right_partner, 1,
        	MPI_COMM_WORLD, MPI_STATUS_IGNORE); // this call handles sending and receiving from right partner proc

          double left1 = H[t][elements_per_proc-1] - H[t][elements_per_proc-2]; 
          double right1 = H[t][elements_per_proc-1] - right_receive;
          double delta1 =  -k*( left1 + right1);
          H[t+1][elements_per_proc-1] = H[t][elements_per_proc-1] + delta1; //computation to determine the value of hte right boundary elements
                                                                            //of middle proc

        }
    	else if ((procid > 0 && procid < total_procs - 1) && (procid % 2 == 1)){ //middle processors and check for if odd and send right first
    		  double SEND_INFO = H[t][elements_per_proc-1]; // value to send to right partner proc
        	right_receive = H[t+1][elements_per_proc-1]; // value of position where receiving will happend when right partner proc sends
        	MPI_Sendrecv(&SEND_INFO, 1, MPI_DOUBLE, right_partner, 1, &right_receive, 1, MPI_DOUBLE, right_partner, 1,
        	MPI_COMM_WORLD, MPI_STATUS_IGNORE); // this call handles sending and receiving from right partner proc

          double left1 = H[t][elements_per_proc-1] - H[t][elements_per_proc-2];
          double right1 = H[t][elements_per_proc-1] - right_receive;
          double delta1 =  -k*( left1 + right1);
          H[t+1][elements_per_proc-1] = H[t][elements_per_proc-1] + delta1;//computation to determine the value of hte right boundary elements
                                                                           //of middle proc
          	
    		  double SEND_INFO1 = H[t][0]; // storing left element
        	left_receive = H[t+1][0]; // value of position where receiving will happen on left hand side of middle proc
        	MPI_Sendrecv(&SEND_INFO1, 1, MPI_DOUBLE, left_partner, 1, &left_receive, 1, MPI_DOUBLE, left_partner, 1,
        	MPI_COMM_WORLD, MPI_STATUS_IGNORE); // this call handles sending and receiving from right partner proc

          double left =  H[t][0] - left_receive;
          double right = H[t][0] - H[t][1];
          double delta = -k*( left + right );
          H[t+1][0] = H[t][0] + delta; //computation to determine the value of hte right boundary elements
                                       //of middle proc
    	}
    	else { //rightmost processors
        	double SEND_INFO = H[t][0]; // the left most element in right most proc
        	right_receive = H[t+1][0]; // value of position where the info will be received from second to last processor
        	MPI_Sendrecv(&SEND_INFO, 1, MPI_DOUBLE, left_partner, 1, &right_receive, 1, MPI_DOUBLE, left_partner, 1,
        	MPI_COMM_WORLD, MPI_STATUS_IGNORE); // this call handles sending the info in send_info into the second to last proc's respective position
          // and receives into the right_receive position from what is being sent from the right of the second to last proc

          double left =  H[t][0] - right_receive;
          double right = H[t][0] - H[t][1];
          double delta = -k*( left + right);
          H[t+1][0] = H[t][0] + delta;//this is where the computation to determine the left most element of
                                      // right most processor happens
    	}
    	//update H[t][p-1] here
    	//computation should happen here, the statements above should only communicate
    }
  }


  if (procid == 0) {
    rbuf = malloc(sizeof(double*) * max_time);
    for (int i = 0; i < max_time; i++) {
      rbuf[i] = malloc(sizeof(double)*elements_per_proc*total_procs); //allocating for all the elements to be received into rbuf
    }
  }

for (int i = 0; i < max_time; i++) {
  double * buffer = NULL;
  if (procid == 0) {
    buffer = rbuf[i];
  }
  MPI_Gather(H[i], elements_per_proc, MPI_DOUBLE, buffer, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //gathers all elements of local arrays from each processor onto root processor
}

  if(print && procid == 0){

    // Column headers
    printf("%3s| ","");
    for(p=0; p<width; p++){
      printf("%5d ",p);
    }
    printf("\n");
    printf("%3s+-","---");
    for(p=0; p<width; p++){
      printf("------");
    }
    printf("\n");
    // Row headers and data
    for(t=0; t<max_time; t++){
      printf("%3d| ",t);
      for(p=0; p<width; p++){
        printf("%5.1f ",rbuf[t][p]);
      }
      printf("\n");
    }
  }

  for(t=0; t<max_time; t++){
    free(H[t]); // freeing each row in H
  }
  free(H);

  if (procid == 0){
    for (int i = 0; i < max_time; i++) {
        free(rbuf[i]); // freeing each row in rbuf
    }
    free(rbuf);
  }
  MPI_Finalize();
  return 0;
}



