#include "nw.h"
#include <string.h>

#define MATCH_SCORE 1
#define MISMATCH_SCORE -1
#define GAP_SCORE -1

#define ALIGN '\\'
#define SKIPA '^'
#define SKIPB '<'

#define MAX(A,B) ( ((A)>(B))?(A):(B) )

#define JOBS_PER_BATCH 256
#define UNROLL_FACTOR 64
#define JOBS_PER_PE ((JOBS_PER_BATCH)/(UNROLL_FACTOR))


void needwun(char SEQA[ALEN], char SEQB[BLEN],
             char alignedA[ALEN+BLEN], char alignedB[ALEN+BLEN]){

    // int M[(ALEN+1)*(BLEN+1)];
    char ptr[(ALEN+1)*(BLEN+1)];

    int M_former[ALEN+1];
    int M_latter[ALEN+1];

    int score, up_left, up, left, max;
    int row, row_up, r;
    int a_idx, b_idx;
    int a_str_idx, b_str_idx;

    // init_row: for(a_idx=0; a_idx<(ALEN+1); a_idx++){
    //     M[a_idx] = a_idx * GAP_SCORE;
    // }
    // init_col: for(b_idx=0; b_idx<(BLEN+1); b_idx++){
    //     M[b_idx*(ALEN+1)] = b_idx * GAP_SCORE;
    // }
    
    // b_idx = 0
    for (a_idx=0; a_idx<ALEN+1; a_idx++) {
        M_former[a_idx] = a_idx*GAP_SCORE;
    }

    // Matrix filling loop
    fill_out: for(b_idx=1; b_idx<(BLEN+1); b_idx++){
	M_latter[0] = M_former[0] + GAP_SCORE;
        fill_in: for(a_idx=1; a_idx<(ALEN+1); a_idx++){
            if(SEQA[a_idx-1] == SEQB[b_idx-1]){
                score = MATCH_SCORE;
            } else {
                score = MISMATCH_SCORE;
            }

            row = (b_idx)*(ALEN+1);

            up_left = M_former[a_idx-1] + score;
            up      = M_former[a_idx  ] + GAP_SCORE;
            left    = M_latter[a_idx-1] + GAP_SCORE;

            max = MAX(up_left, MAX(up, left));

            M_latter[a_idx] = max;
            if(max == left){
                ptr[row + a_idx] = SKIPB;
            } else if(max == up){
                ptr[row + a_idx] = SKIPA;
            } else{
                ptr[row + a_idx] = ALIGN;
            }
        }
	for (int k=0; k<ALEN+1; k++) {
	    M_former[k] = M_latter[k];
	}
    }

    // TraceBack (n.b. aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    trace: while(a_idx>0 || b_idx>0) {
        r = b_idx*(ALEN+1);
        if (ptr[r + a_idx] == ALIGN){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            a_idx--;
            b_idx--;
        }
        else if (ptr[r + a_idx] == SKIPB){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = '-';
            a_idx--;
        }
        else{ // SKIPA
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            b_idx--;
        }
    }

    // Pad the result
    pad_a: for( ; a_str_idx<ALEN+BLEN; a_str_idx++ ) {
      alignedA[a_str_idx] = '_';
    }
    pad_b: for( ; b_str_idx<ALEN+BLEN; b_str_idx++ ) {
      alignedB[b_str_idx] = '_';
    }
}

void needwun_tiling(char* SEQA, char* SEQB,
             char* alignedA, char* alignedB) {
	for (int i=0; i<JOBS_PER_PE; i++) {
	    needwun(SEQA + i*ALEN, SEQB + i*BLEN,
		    alignedA + i*(ALEN+BLEN), alignedB + i*(ALEN+BLEN));
	}
	return;
}

void buffer_load(int flag, char* global_buf_A, char part_buf_A[UNROLL_FACTOR][ALEN*JOBS_PER_PE], char* global_buf_B, char part_buf_B[UNROLL_FACTOR][BLEN*JOBS_PER_PE]) {
#pragma HLS INLINE off
  if (flag) {
    for (int i=0; i<UNROLL_FACTOR; i++) {
      memcpy(part_buf_A[i], global_buf_A + i * (ALEN*JOBS_PER_PE), ALEN*JOBS_PER_PE);
      memcpy(part_buf_B[i], global_buf_B + i * (BLEN*JOBS_PER_PE), BLEN*JOBS_PER_PE);
    }
  }
  return;
}

void buffer_store(int flag, char* global_buf_A, char part_buf_A[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE], char* global_buf_B, char part_buf_B[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE]) {
#pragma HLS INLINE off
  if (flag) {
    for (int i=0; i<UNROLL_FACTOR; i++) {
      memcpy(global_buf_A + i * ((ALEN+BLEN)*JOBS_PER_PE), part_buf_A[i], (ALEN+BLEN)*JOBS_PER_PE);
      memcpy(global_buf_B + i * ((ALEN+BLEN)*JOBS_PER_PE), part_buf_B[i], (ALEN+BLEN)*JOBS_PER_PE);
    }
  }
  return;
}

void buffer_compute(int flag, char seqA_buf[UNROLL_FACTOR][ALEN*JOBS_PER_PE],
	                      char seqB_buf[UNROLL_FACTOR][BLEN*JOBS_PER_PE],
		              char alignedA_buf[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE],      
                              char alignedB_buf[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE]) {
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j=0; j<UNROLL_FACTOR; j++) {
    #pragma HLS UNROLL
	alignedA_buf[j][0] = seqA_buf[j][0];
	alignedB_buf[j][0] = seqB_buf[j][0];
	//needwun_tiling(seqA_buf[j], seqB_buf[j], alignedA_buf[j], alignedB_buf[j]);
    }
  }
  return;
}

void workload(char* SEQA, char* SEQB,
             char* alignedA, char* alignedB, int num_jobs) {
#pragma HLS INTERFACE m_axi port=SEQA offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=SEQB offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=alignedA offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=alignedB offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=SEQA bundle=control
#pragma HLS INTERFACE s_axilite port=SEQB bundle=control
#pragma HLS INTERFACE s_axilite port=alignedA bundle=control
#pragma HLS INTERFACE s_axilite port=alignedB bundle=control
#pragma HLS INTERFACE s_axilite port=num_jobs bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  int num_batches = num_jobs / JOBS_PER_BATCH;

  char seqA_buf_x[UNROLL_FACTOR][ALEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqA_buf_x cyclic factor=64 dim=1
  char seqA_buf_y[UNROLL_FACTOR][ALEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqA_buf_y cyclic factor=64 dim=1
  char seqA_buf_z[UNROLL_FACTOR][ALEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqA_buf_z cyclic factor=64 dim=1

  char seqB_buf_x[UNROLL_FACTOR][BLEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqB_buf_x cyclic factor=64 dim=1
  char seqB_buf_y[UNROLL_FACTOR][BLEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqB_buf_y cyclic factor=64 dim=1
  char seqB_buf_z[UNROLL_FACTOR][BLEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqB_buf_z cyclic factor=64 dim=1

  char alignedA_buf_x[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedA_buf_x cyclic factor=64 dim=1
  char alignedA_buf_y[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedA_buf_y cyclic factor=64 dim=1
  char alignedA_buf_z[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedA_buf_z cyclic factor=64 dim=1

  char alignedB_buf_x[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedB_buf_x cyclic factor=64 dim=1
  char alignedB_buf_y[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedB_buf_y cyclic factor=64 dim=1
  char alignedB_buf_z[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedB_buf_z cyclic factor=64 dim=1

  int i;
  for (i=0; i<num_batches; i++) {
    buffer_load(1, SEQA+i*ALEN*JOBS_PER_BATCH, seqA_buf_x, SEQB+i*BLEN*JOBS_PER_BATCH, seqB_buf_x);
    buffer_compute(1, seqA_buf_x, seqB_buf_x, alignedA_buf_x, alignedB_buf_x);
    buffer_store(1, alignedA+(i)*(ALEN+BLEN)*JOBS_PER_BATCH, alignedA_buf_x, alignedB+(i)*(ALEN+BLEN)*JOBS_PER_BATCH, alignedB_buf_x);
  }
  return;
}
