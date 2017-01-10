#include "ap_int.h"
#define RANGE(var, h, l) (var).range(h, l)
/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/
#include "spmv.h"
#include <string.h>
#define ROWS_PER_TILE 256
#define UNROLL_FACTOR 16
extern "C" {

void ellpack(ap_uint<64> *nzval,ap_uint<64> *cols,ap_uint<64> *vec,ap_uint<64> *out)
{
  
#pragma HLS INLINE off
  int i;
  int j;
  double Si;
  double sum;
  for (i = 0; i < 256 / 16; i++) {
    
#pragma HLS PIPELINE
    double fpTmp5 = 0.0;
    RANGE(out[i / 1],i % 1 * 64 + 63,i % 1 * 64) =  *((long *)(&fpTmp5));
  }
  ellpack_2:
  for (j = 0; j < 512; j++) {
    
#pragma HLS PIPELINE
    ellpack_1:
    for (i = 0; i < 256 / 16; i++) {
      
#pragma HLS UNROLL
      short tmp = RANGE(cols[(j + i * 512) / 4],(j + i * 512) % 4 * 16 + 15,(j + i * 512) % 4 * 16);
      long uintTmp2 = RANGE(out[i / 1],i % 1 * 64 + 63,i % 1 * 64);
      double castTmp2 =  *((double *)(&uintTmp2));
      long uintTmp3 = RANGE(nzval[(j + i * 512) / 1],(j + i * 512) % 1 * 64 + 63,(j + i * 512) % 1 * 64);
      double castTmp3 =  *((double *)(&uintTmp3));
      long uintTmp4 = RANGE(vec[tmp / 1],tmp % 1 * 64 + 63,tmp % 1 * 64);
      double castTmp4 =  *((double *)(&uintTmp4));
      double fpTmp6 = castTmp2 + castTmp3 * castTmp4;
      RANGE(out[i / 1],i % 1 * 64 + 63,i % 1 * 64) =  *((long *)(&fpTmp6));
    }
  }
}

void load_nzval(ap_uint<64> *nzval,ap_uint<64> local_nzval[16UL][8192UL],int flag)
{
  
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j = 0; j < 16; j++) {
      memcpy(local_nzval[j],(nzval + j * 256 / 16 * 512 / 1),sizeof(double ) * 256 * 512 / 16);
    }
  }
}

void load_cols(ap_uint<64> *cols,ap_uint<64> local_cols[16UL][2048UL],int flag)
{
  
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j = 0; j < 16; j++) {
      memcpy(local_cols[j],(cols + j * 256 / 16 * 512 / 4),sizeof(short ) * 256 * 512 / 16);
    }
  }
}

void buffer_compute(ap_uint<64> local_nzval[16UL][8192UL],ap_uint<64> local_cols[16UL][2048UL],ap_uint<64> local_vec[16UL][4096UL],ap_uint<64> local_out[16UL][16UL],int flag,ap_uint<64> *out)
{
  
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j = 0; j < 16; j++) {
      
#pragma HLS UNROLL
      ellpack(local_nzval[j],local_cols[j],local_vec[j],local_out[j]);
    }
    for (j = 0; j < 16; j++) {
      memcpy((out + j * 256 / 16 / 1),local_out[j],sizeof(double ) * 256 / 16);
    }
  }
}

void workload(ap_uint<64> *nzval,ap_uint<64> *cols,ap_uint<64> *vec,ap_uint<64> *out)
{
  
#pragma HLS INTERFACE m_axi port=nzval offset=slave bundle=gmem1
  
#pragma HLS INTERFACE m_axi port=cols offset=slave bundle=gmem2
  
#pragma HLS INTERFACE m_axi port=vec offset=slave bundle=gmem1
  
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1
  
#pragma HLS INTERFACE s_axilite port=nzval bundle=control
  
#pragma HLS INTERFACE s_axilite port=cols bundle=control
  
#pragma HLS INTERFACE s_axilite port=vec bundle=control
  
#pragma HLS INTERFACE s_axilite port=out bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  int num_tiles = 4096 / 256;
  int i;
  int j;
  int k;
  ap_uint<64> local_nzval_x[16UL][8192UL];
  
#pragma HLS ARRAY_PARTITION variable=local_nzval_x dim=1 complete
  ap_uint<64> local_cols_x[16UL][2048UL];
  
#pragma HLS ARRAY_PARTITION variable=local_cols_x dim=1 complete
  ap_uint<64> local_nzval_y[16UL][8192UL];
  
#pragma HLS ARRAY_PARTITION variable=local_nzval_y dim=1 complete
  ap_uint<64> local_cols_y[16UL][2048UL];
  
#pragma HLS ARRAY_PARTITION variable=local_cols_y dim=1 complete
  ap_uint<64> local_vec[16UL][4096UL];
  
#pragma HLS ARRAY_PARTITION variable=local_vec dim=1 complete
  memcpy(local_vec[0],vec,4096 * sizeof(double ));
  for (i = 0; i < 4096; i++) {
    
#pragma HLS PIPELINE
    for (j = 1; j < 16; j++) {
      
#pragma HLS UNROLL
      long uintTmp0 = RANGE(local_vec[0][i / 1],i % 1 * 64 + 63,i % 1 * 64);
      double castTmp0 =  *((double *)(&uintTmp0));
      double fpTmp1 = castTmp0;
      RANGE(local_vec[j][i / 1],i % 1 * 64 + 63,i % 1 * 64) =  *((long *)(&fpTmp1));
    }
  }
  ap_uint<64> local_out[16UL][16UL];
  
#pragma HLS ARRAY_PARTITION variable=local_out dim=1 complete
  int load_flag;
  int compute_flag;
  for (i = 0; i < num_tiles + 1; i++) {
    load_flag = (i >= 0 && i < num_tiles);
    compute_flag = (i > 0 && i <= num_tiles);
    if (i % 2 == 0) {
      load_nzval(nzval + i * 256 * 512 / 1,local_nzval_x,load_flag);
      load_cols(cols + i * 256 * 512 / 4,local_cols_x,load_flag);
      buffer_compute(local_nzval_y,local_cols_y,local_vec,local_out,compute_flag,out + (i - 1) * 256 / 1);
    }
     else {
      load_nzval(nzval + i * 256 * 512 / 1,local_nzval_y,load_flag);
      load_cols(cols + i * 256 * 512 / 4,local_cols_y,load_flag);
      buffer_compute(local_nzval_x,local_cols_x,local_vec,local_out,compute_flag,out + (i - 1) * 256 / 1);
    }
  }
  return ;
}
}
