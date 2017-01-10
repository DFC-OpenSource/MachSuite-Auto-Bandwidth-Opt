#include "ap_int.h"
#define RANGE(var, h, l) (var).range(h, l)
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
#define UNROLL_FACTOR 32
#define JOBS_PER_PE ((JOBS_PER_BATCH)/(UNROLL_FACTOR))
extern "C" {

void needwun(ap_uint<16> SEQA[64UL],ap_uint<16> SEQB[64UL],ap_uint<16> alignedA[128UL],ap_uint<16> alignedB[128UL])
{
  ap_uint<16> ptr[8321UL];
  ap_uint<16> M_former[65UL];
  
#pragma HLS ARRAY_PARTITION variable=M_former dim=0 complete
  ap_uint<16> M_latter[65UL];
  
#pragma HLS ARRAY_PARTITION variable=M_latter dim=0 complete
  char score;
  char up_left;
  char up;
  char left;
  char max;
  int row;
  int row_up;
  int r;
  int a_idx;
  int b_idx;
  int a_str_idx;
  int b_str_idx;
  for (a_idx = 0; a_idx < 128 + 1; a_idx++) {
    
#pragma HLS UNROLL
    RANGE(M_former[a_idx / 2],a_idx % 2 * 8 + 7,a_idx % 2 * 8) = (a_idx * - 1);
  }
// Matrix filling loop
  fill_out:
  for (b_idx = 1; b_idx < 128 + 1; b_idx++) {
    fill_in:
    for (a_idx = 0; a_idx < 128 + 1; a_idx++) {
      
#pragma HLS PIPELINE
      if (a_idx == 0) {
        RANGE(M_latter[0 / 2],0 % 2 * 8 + 7,0 % 2 * 8) = (b_idx * - 1);
      }
       else {
        if ((RANGE(SEQA[(a_idx - 1) / 2],(a_idx - 1) % 2 * 8 + 7,(a_idx - 1) % 2 * 8)) == (RANGE(SEQB[(b_idx - 1) / 2],(b_idx - 1) % 2 * 8 + 7,(b_idx - 1) % 2 * 8))) {
          score = 1;
        }
         else {
          score = (- 1);
        }
        char x = RANGE(M_former[128 / 2],128 % 2 * 8 + 7,128 % 2 * 8);
        char y = RANGE(M_former[0 / 2],0 % 2 * 8 + 7,0 % 2 * 8);
        char z = RANGE(M_latter[128 / 2],128 % 2 * 8 + 7,128 % 2 * 8);
        up_left = (x + score);
        up = (y + - 1);
        left = (z + - 1);
        max = (up_left > ((up > left?up : left))?up_left : ((up > left?up : left)));
        RANGE(M_latter[0 / 2],0 % 2 * 8 + 7,0 % 2 * 8) = max;
        row = b_idx * (128 + 1);
        if (max == left) {
          RANGE(ptr[(row + a_idx) / 2],(row + a_idx) % 2 * 8 + 7,(row + a_idx) % 2 * 8) = '<';
        }
         else if (max == up) {
          RANGE(ptr[(row + a_idx) / 2],(row + a_idx) % 2 * 8 + 7,(row + a_idx) % 2 * 8) = '^';
        }
         else {
          RANGE(ptr[(row + a_idx) / 2],(row + a_idx) % 2 * 8 + 7,(row + a_idx) % 2 * 8) = '\\';
        }
      }
//-- shifting register
      char tmp_former = RANGE(M_former[0 / 2],0 % 2 * 8 + 7,0 % 2 * 8);
      char tmp_latter = RANGE(M_latter[0 / 2],0 % 2 * 8 + 7,0 % 2 * 8);
      for (int i = 0; i < 128 + 1 - 1; i++) {
        RANGE(M_former[i / 2],i % 2 * 8 + 7,i % 2 * 8) = RANGE(M_former[(i + 1) / 2],(i + 1) % 2 * 8 + 7,(i + 1) % 2 * 8);
        RANGE(M_latter[i / 2],i % 2 * 8 + 7,i % 2 * 8) = RANGE(M_latter[(i + 1) / 2],(i + 1) % 2 * 8 + 7,(i + 1) % 2 * 8);
      }
      RANGE(M_former[(128 + 1 - 1) / 2],(128 + 1 - 1) % 2 * 8 + 7,(128 + 1 - 1) % 2 * 8) = tmp_former;
      RANGE(M_latter[(128 + 1 - 1) / 2],(128 + 1 - 1) % 2 * 8 + 7,(128 + 1 - 1) % 2 * 8) = tmp_latter;
    }
    for (int k = 0; k < 128 + 1; k++) {
      
#pragma HLS UNROLL
      RANGE(M_former[k / 2],k % 2 * 8 + 7,k % 2 * 8) = RANGE(M_latter[k / 2],k % 2 * 8 + 7,k % 2 * 8);
    }
  }
// TraceBack (n.b. aligned sequences are backwards to avoid string appending)
  a_idx = 128;
  b_idx = 128;
  a_str_idx = 0;
  b_str_idx = 0;
  trace:
  while(a_idx > 0 && b_idx > 0){
//trace: while(a_idx>0 || b_idx>0) {
    r = b_idx * (128 + 1);
    if ((RANGE(ptr[(r + a_idx) / 2],(r + a_idx) % 2 * 8 + 7,(r + a_idx) % 2 * 8)) == '\\') {
      RANGE(alignedA[a_str_idx / 2],a_str_idx % 2 * 8 + 7,a_str_idx % 2 * 8) = RANGE(SEQA[(a_idx - 1) / 2],(a_idx - 1) % 2 * 8 + 7,(a_idx - 1) % 2 * 8);
      a_str_idx++;
      RANGE(alignedB[b_str_idx / 2],b_str_idx % 2 * 8 + 7,b_str_idx % 2 * 8) = RANGE(SEQB[(b_idx - 1) / 2],(b_idx - 1) % 2 * 8 + 7,(b_idx - 1) % 2 * 8);
      b_str_idx++;
      a_idx--;
      b_idx--;
    }
     else if ((RANGE(ptr[(r + a_idx) / 2],(r + a_idx) % 2 * 8 + 7,(r + a_idx) % 2 * 8)) == '<') {
      RANGE(alignedA[a_str_idx / 2],a_str_idx % 2 * 8 + 7,a_str_idx % 2 * 8) = RANGE(SEQA[(a_idx - 1) / 2],(a_idx - 1) % 2 * 8 + 7,(a_idx - 1) % 2 * 8);
      a_str_idx++;
      RANGE(alignedB[b_str_idx / 2],b_str_idx % 2 * 8 + 7,b_str_idx % 2 * 8) = '-';
      b_str_idx++;
      a_idx--;
    }
     else 
// SKIPA
{
      RANGE(alignedA[a_str_idx / 2],a_str_idx % 2 * 8 + 7,a_str_idx % 2 * 8) = '-';
      a_str_idx++;
      RANGE(alignedB[b_str_idx / 2],b_str_idx % 2 * 8 + 7,b_str_idx % 2 * 8) = RANGE(SEQB[(b_idx - 1) / 2],(b_idx - 1) % 2 * 8 + 7,(b_idx - 1) % 2 * 8);
      b_str_idx++;
      b_idx--;
    }
  }
// Pad the result
  pad_a:
  for (; a_str_idx < 128 + 128; a_str_idx++) {
    RANGE(alignedA[a_str_idx / 2],a_str_idx % 2 * 8 + 7,a_str_idx % 2 * 8) = '_';
  }
  pad_b:
  for (; b_str_idx < 128 + 128; b_str_idx++) {
    RANGE(alignedB[b_str_idx / 2],b_str_idx % 2 * 8 + 7,b_str_idx % 2 * 8) = '_';
  }
}

void needwun_tiling(ap_uint<16> *SEQA,ap_uint<16> *SEQB,ap_uint<16> *alignedA,ap_uint<16> *alignedB)
{
  for (int i = 0; i < 256 / 32; i++) {
    needwun(SEQA + i * 128 / 2,SEQB + i * 128 / 2,alignedA + i * (128 + 128) / 2,alignedB + i * (128 + 128) / 2);
  }
  return ;
}

void buffer_load(int flag,ap_uint<16> *global_buf_A,ap_uint<16> part_buf_A[32UL][512UL],ap_uint<16> *global_buf_B,ap_uint<16> part_buf_B[32UL][512UL])
{
  
#pragma HLS INLINE off
  if (flag) {
    for (int i = 0; i < 32; i++) {
      memcpy(part_buf_A[i],(global_buf_A + i * (128 * (256 / 32)) / 2),(128 * (256 / 32)));
      memcpy(part_buf_B[i],(global_buf_B + i * (128 * (256 / 32)) / 2),(128 * (256 / 32)));
    }
  }
  return ;
}

void buffer_store(int flag,ap_uint<16> *global_buf_A,ap_uint<16> part_buf_A[32UL][1024UL],ap_uint<16> *global_buf_B,ap_uint<16> part_buf_B[32UL][1024UL])
{
  
#pragma HLS INLINE off
  if (flag) {
    for (int i = 0; i < 32; i++) {
      memcpy((global_buf_A + i * ((128 + 128) * (256 / 32)) / 2),part_buf_A[i],((128 + 128) * (256 / 32)));
      memcpy((global_buf_B + i * ((128 + 128) * (256 / 32)) / 2),part_buf_B[i],((128 + 128) * (256 / 32)));
    }
  }
  return ;
}

void buffer_compute(int flag,ap_uint<16> seqA_buf[32UL][512UL],ap_uint<16> seqB_buf[32UL][512UL],ap_uint<16> alignedA_buf[32UL][1024UL],ap_uint<16> alignedB_buf[32UL][1024UL])
{
  
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j = 0; j < 32; j++) {
      
#pragma HLS UNROLL
      needwun_tiling(seqA_buf[j],seqB_buf[j],alignedA_buf[j],alignedB_buf[j]);
    }
  }
  return ;
}

void workload(ap_uint<16> *SEQA,ap_uint<16> *SEQB,ap_uint<16> *alignedA,ap_uint<16> *alignedB,int num_jobs)
{
  
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
  int num_batches = num_jobs / 256;
  ap_uint<16> seqA_buf_x[32UL][512UL];
  
#pragma HLS ARRAY_PARTITION variable=seqA_buf_x complete dim=1
  ap_uint<16> seqA_buf_y[32UL][512UL];
  
#pragma HLS ARRAY_PARTITION variable=seqA_buf_y complete dim=1
  ap_uint<16> seqA_buf_z[32UL][512UL];
  
#pragma HLS ARRAY_PARTITION variable=seqA_buf_z complete dim=1
  ap_uint<16> seqB_buf_x[32UL][512UL];
  
#pragma HLS ARRAY_PARTITION variable=seqB_buf_x complete dim=1
  ap_uint<16> seqB_buf_y[32UL][512UL];
  
#pragma HLS ARRAY_PARTITION variable=seqB_buf_y complete dim=1
  ap_uint<16> seqB_buf_z[32UL][512UL];
  
#pragma HLS ARRAY_PARTITION variable=seqB_buf_z complete dim=1
  ap_uint<16> alignedA_buf_x[32UL][1024UL];
  
#pragma HLS ARRAY_PARTITION variable=alignedA_buf_x complete dim=1
  ap_uint<16> alignedA_buf_y[32UL][1024UL];
  
#pragma HLS ARRAY_PARTITION variable=alignedA_buf_y complete dim=1
  ap_uint<16> alignedA_buf_z[32UL][1024UL];
  
#pragma HLS ARRAY_PARTITION variable=alignedA_buf_z complete dim=1
  ap_uint<16> alignedB_buf_x[32UL][1024UL];
  
#pragma HLS ARRAY_PARTITION variable=alignedB_buf_x complete dim=1
  ap_uint<16> alignedB_buf_y[32UL][1024UL];
  
#pragma HLS ARRAY_PARTITION variable=alignedB_buf_y complete dim=1
  ap_uint<16> alignedB_buf_z[32UL][1024UL];
  
#pragma HLS ARRAY_PARTITION variable=alignedB_buf_z complete dim=1
  int i;
  for (i = 0; i < num_batches + 2; i++) {
    int load_flag = (i >= 0 && i < num_batches);
    int compute_flag = (i >= 1 && i < num_batches + 1);
    int store_flag = (i >= 2 && i < num_batches + 2);
    if (i % 3 == 0) {
      buffer_load(load_flag,SEQA + i * 128 * 256 / 2,seqA_buf_x,SEQB + i * 128 * 256 / 2,seqB_buf_x);
      buffer_compute(compute_flag,seqA_buf_z,seqB_buf_z,alignedA_buf_z,alignedB_buf_z);
      buffer_store(store_flag,alignedA + (i - 2) * (128 + 128) * 256 / 2,alignedA_buf_y,alignedB + (i - 2) * (128 + 128) * 256 / 2,alignedB_buf_y);
    }
     else if (i % 3 == 1) {
      buffer_load(load_flag,SEQA + i * 128 * 256 / 2,seqA_buf_y,SEQB + i * 128 * 256 / 2,seqB_buf_y);
      buffer_compute(compute_flag,seqA_buf_x,seqB_buf_x,alignedA_buf_x,alignedB_buf_x);
      buffer_store(store_flag,alignedA + (i - 2) * (128 + 128) * 256 / 2,alignedA_buf_z,alignedB + (i - 2) * (128 + 128) * 256 / 2,alignedB_buf_z);
    }
     else {
      buffer_load(load_flag,SEQA + i * 128 * 256 / 2,seqA_buf_z,SEQB + i * 128 * 256 / 2,seqB_buf_z);
      buffer_compute(compute_flag,seqA_buf_y,seqB_buf_y,alignedA_buf_y,alignedB_buf_y);
      buffer_store(store_flag,alignedA + (i - 2) * (128 + 128) * 256 / 2,alignedA_buf_x,alignedB + (i - 2) * (128 + 128) * 256 / 2,alignedB_buf_x);
    }
  }
  return ;
}
}
