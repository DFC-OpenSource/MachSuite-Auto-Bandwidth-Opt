#include "viterbi.h"
#include <string.h>
#include <inttypes.h>

#define JOBS_PER_BATCH 256
#define UNROLL_FACTOR 128
#define JOBS_PER_PE ((JOBS_PER_BATCH)/(UNROLL_FACTOR))

extern "C" {

int viterbi( uint8_t obs[N_OBS], TYPE init[N_STATES], TYPE transition[N_STATES*N_STATES], TYPE emission[N_STATES*N_TOKENS], uint8_t path[N_OBS] )
{
  TYPE llike[N_OBS][N_STATES];
  uint32_t t;
  uint8_t prev, curr;
  TYPE min_p, p;
  uint8_t min_s, s;
  // All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) )
 
  // Initialize with first observation and initial probabilities
  L_init: for( s=0; s<N_STATES; s++ ) {
  #pragma HLS PIPELINE
		uint8_t tmp = obs[0];
    llike[0][s] = init[s] + emission[s*N_TOKENS+tmp];
  }

  // Iteratively compute the probabilities over time
  L_timestep: for( t=1; t<N_OBS; t++ ) {
    L_curr_state: for( curr=0; curr<N_STATES; curr++ ) {
      // Compute likelihood HMM is in current state and where it came from.
      L_prev_state: for( prev=0; prev<N_STATES; prev++ ) {
      #pragma HLS PIPELINE
				uint8_t tmp = obs[t];
        p = llike[t-1][prev] +
            transition[prev*N_STATES+curr] +
            emission[curr*N_TOKENS+tmp];
	if (!prev || p < min_p) {
	  min_p = p;
	}
        llike[t][curr] = min_p;
      }
    }
  }

  // Identify end state
  min_s = 0;
  min_p = llike[N_OBS-1][min_s];
  L_end: for( s=1; s<N_STATES; s++ ) {
  #pragma HLS PIPELINE
    p = llike[N_OBS-1][s];
    if( p<min_p ) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  // Backtrack to recover full path
  L_backtrack: for( t=N_OBS-2; t>=0; t-- ) {
    L_state: for( s=0; s<N_STATES; s++ ) {
    #pragma HLS PIPELINE II=2
			uint8_t tmp = path[t+1];
      p = llike[t][s] + transition[s*N_STATES+tmp];
      if (!s || p < min_p) {
        min_p = p;
        min_s = s;
      }
    }
    path[t] = min_s;
  }

  return 0;
}

void viterbi_tiling( uint8_t* obs, TYPE* init, TYPE* transition, TYPE* emission, uint8_t* path ) {
    for (int j=0; j<JOBS_PER_PE; j++) {
        viterbi(obs + j*N_OBS, init + j*N_STATES, transition + j*TRANS_SIZE, emission + j*EMIT_SIZE, path + j*N_OBS);
    }
}

void buffer_load(int flag,  uint8_t* global_buf_A,  uint8_t part_buf_A[UNROLL_FACTOR][N_OBS*JOBS_PER_PE],
		           TYPE* global_buf_B, TYPE part_buf_B[UNROLL_FACTOR][N_STATES*JOBS_PER_PE],
		           TYPE* global_buf_C, TYPE part_buf_C[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE],
		           TYPE* global_buf_D, TYPE part_buf_D[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE]
		) {
#pragma HLS INLINE off
  int i;
  if (flag) {
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_A[i], global_buf_A + i * (N_OBS*JOBS_PER_PE), sizeof(uint8_t)*N_OBS*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_B[i], global_buf_B + i * (N_STATES*JOBS_PER_PE), sizeof(TYPE)*N_STATES*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_C[i], global_buf_C + i * (TRANS_SIZE*JOBS_PER_PE), sizeof(TYPE)*TRANS_SIZE*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_D[i], global_buf_D + i * (EMIT_SIZE*JOBS_PER_PE), sizeof(TYPE)*EMIT_SIZE*JOBS_PER_PE);
    }
  }
  return;
}

void buffer_store(int flag, uint8_t* global_buf_A, uint8_t part_buf_A[UNROLL_FACTOR][N_OBS*JOBS_PER_PE]) {
#pragma HLS INLINE off
  if (flag) {
    for (int i=0; i<UNROLL_FACTOR; i++) {
      memcpy(global_buf_A + i * (N_OBS*JOBS_PER_PE), part_buf_A[i], sizeof(uint8_t)*N_OBS*JOBS_PER_PE);
    }
  }
  return;
}

void buffer_compute(int flag,  uint8_t local_obs[UNROLL_FACTOR][N_OBS*JOBS_PER_PE],
	                      TYPE local_init[UNROLL_FACTOR][N_STATES*JOBS_PER_PE],
		              TYPE local_transition[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE],      
                              TYPE local_emission[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE],
                             uint8_t local_path[UNROLL_FACTOR][N_OBS*JOBS_PER_PE]) {
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j=0; j<UNROLL_FACTOR; j++) {
    #pragma HLS UNROLL
	viterbi_tiling(local_obs[j], local_init[j], local_transition[j], local_emission[j], local_path[j]);
    }
  }
  return;
}

void workload( uint8_t* obs, TYPE* init, TYPE* transition, TYPE* emission, uint8_t* path, int num_jobs ) {
#pragma HLS INTERFACE m_axi port=obs offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=init offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=transition offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=emission offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=obs bundle=control
#pragma HLS INTERFACE s_axilite port=init bundle=control
#pragma HLS INTERFACE s_axilite port=transition bundle=control
#pragma HLS INTERFACE s_axilite port=emission bundle=control
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=num_jobs bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    uint8_t local_obs_x[UNROLL_FACTOR][N_OBS*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_obs_x complete dim=1
    uint8_t local_obs_y[UNROLL_FACTOR][N_OBS*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_obs_y complete dim=1

    TYPE local_init_x[UNROLL_FACTOR][N_STATES*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_init_x complete dim=1
    TYPE local_init_y[UNROLL_FACTOR][N_STATES*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_init_y complete dim=1

    TYPE local_transition_x[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_transition_x complete dim=1
    TYPE local_transition_y[UNROLL_FACTOR][TRANS_SIZE*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_transition_y complete dim=1

    TYPE local_emission_x[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_emission_x complete dim=1
    TYPE local_emission_y[UNROLL_FACTOR][EMIT_SIZE*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_emission_y complete dim=1

    uint8_t local_path_x[UNROLL_FACTOR][N_OBS*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_path_x complete dim=1
    uint8_t local_path_y[UNROLL_FACTOR][N_OBS*JOBS_PER_PE];
    #pragma HLS ARRAY_PARTITION variable=local_path_y complete dim=1


    int num_batches = num_jobs / JOBS_PER_BATCH;

    int i;
    for (i=0; i<num_batches+2; i++) {
      int load_flag = i >= 0 && i < num_batches;
      int compute_flag = i >= 1 && i < num_batches+1;
      int store_flag = i >= 2 && i < num_batches+2;
      if (i % 2 == 0) {
        buffer_load(load_flag, obs+i*JOBS_PER_BATCH*N_OBS, local_obs_x, init+i*JOBS_PER_BATCH*N_STATES, local_init_x, transition+i*JOBS_PER_BATCH*TRANS_SIZE, local_transition_x, emission+i*JOBS_PER_BATCH*EMIT_SIZE, local_emission_x);
        buffer_compute(compute_flag, local_obs_y, local_init_y, local_transition_y, local_emission_y, local_path_y);
        buffer_store(store_flag, path+(i-2)*JOBS_PER_BATCH, local_path_x);
      } 
      else {
        buffer_load(load_flag, obs+i*JOBS_PER_BATCH*N_OBS, local_obs_y, init+i*JOBS_PER_BATCH*N_STATES, local_init_y, transition+i*JOBS_PER_BATCH*TRANS_SIZE, local_transition_y, emission+i*JOBS_PER_BATCH*EMIT_SIZE, local_emission_y);
        buffer_compute(compute_flag, local_obs_x, local_init_x, local_transition_x, local_emission_x, local_path_x);
        buffer_store(store_flag, path+(i-2)*JOBS_PER_BATCH, local_path_y);
      } 
    }
    return;
}

}
