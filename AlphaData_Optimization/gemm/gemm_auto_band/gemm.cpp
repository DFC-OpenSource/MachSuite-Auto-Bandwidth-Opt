#include "ap_int.h"
#define RANGE(var, h, l) (var).range(h, l)
#include "gemm.h"
extern "C" {

void load(int flag,int i,int j,int k,ap_uint<512> local_ma[128UL][16UL],ap_uint<512> local_mb[128UL][16UL],ap_uint<512> ma[131072UL],ap_uint<512> mb[131072UL])
{
  
#pragma HLS INLINE off
  int ii;
  int jj;
  int kk;
  if (flag) {
    for (ii = 0; ii < 128; ii++) {
      for (kk = 0; kk < 128; kk++) {
        
#pragma HLS PIPELINE II = 1
        long uintTmp3 = RANGE(ma[((i + ii) * 1024 + (k + kk)) / 8],((i + ii) * 1024 + (k + kk)) % 8 * 64 + 63,((i + ii) * 1024 + (k + kk)) % 8 * 64);
        double castTmp3 =  *((double *)(&uintTmp3));
        double fpTmp5 = castTmp3;
        RANGE(local_ma[ii][kk / 8],kk % 8 * 64 + 63,kk % 8 * 64) =  *((long *)(&fpTmp5));
      }
    }
    for (kk = 0; kk < 128; kk++) {
      for (jj = 0; jj < 128; jj++) {
        
#pragma HLS PIPELINE II = 1
        long uintTmp4 = RANGE(mb[((k + kk) * 1024 + (j + jj)) / 8],((k + kk) * 1024 + (j + jj)) % 8 * 64 + 63,((k + kk) * 1024 + (j + jj)) % 8 * 64);
        double castTmp4 =  *((double *)(&uintTmp4));
        double fpTmp6 = castTmp4;
        RANGE(local_mb[kk][jj / 8],jj % 8 * 64 + 63,jj % 8 * 64) =  *((long *)(&fpTmp6));
      }
    }
  }
}

void compute(int flag,ap_uint<512> local_ma[128UL][16UL],ap_uint<512> local_mb[128UL][16UL],ap_uint<512> local_prod[128UL][16UL])
{
  
#pragma HLS INLINE off
  int ii;
  int jj;
  int kk;
  int uu;
  if (flag) {
    for (kk = 0; kk < 128; kk++) {
      for (ii = 0; ii < 128; ii++) {
        for (jj = 0; jj < 128; jj += 64) {
          
#pragma HLS PIPELINE II = 1
          for (uu = 0; uu < 64; uu++) {
            
#pragma HLS UNROLL
            
#pragma HLS DEPENDENCE variable="local_prod" inter false
            long uintTmp7 = RANGE(local_ma[ii][kk / 8],kk % 8 * 64 + 63,kk % 8 * 64);
            double castTmp7 =  *((double *)(&uintTmp7));
            long uintTmp8 = RANGE(local_mb[kk][(jj + uu) / 8],(jj + uu) % 8 * 64 + 63,(jj + uu) % 8 * 64);
            double castTmp8 =  *((double *)(&uintTmp8));
            double fpTmp9 = castTmp7 * castTmp8;
            RANGE(local_prod[ii][(jj + uu) / 8],(jj + uu) % 8 * 64 + 63,(jj + uu) % 8 * 64) +=  *((long *)(&fpTmp9));
          }
        }
      }
    }
  }
}

void workload(ap_uint<512> ma[131072UL],ap_uint<512> mb[131072UL],ap_uint<512> prod[131072UL])
{
  
#pragma HLS INTERFACE m_axi port=ma offset=slave bundle=gmem
  
#pragma HLS INTERFACE m_axi port=mb offset=slave bundle=gmem
  
#pragma HLS INTERFACE m_axi port=prod offset=slave bundle=gmem
  
#pragma HLS INTERFACE s_axilite port=ma bundle=control
  
#pragma HLS INTERFACE s_axilite port=mb bundle=control
  
#pragma HLS INTERFACE s_axilite port=prod bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  int d;
  int f;
  ap_uint<512> local_ma_ping[128UL][16UL];
  ap_uint<512> local_mb_ping[128UL][16UL];
  
#pragma HLS ARRAY_PARTITION variable=local_mb_ping complete dim=2
  ap_uint<512> local_ma_pong[128UL][16UL];
  ap_uint<512> local_mb_pong[128UL][16UL];
  
#pragma HLS ARRAY_PARTITION variable=local_mb_pong complete dim=2
  ap_uint<512> local_prod[128UL][16UL];
  
#pragma HLS ARRAY_PARTITION variable=local_prod complete dim=2
//array partition
  double mult;
  double sum;
  int i;
  int j;
  int k;
  int ii;
  int jj;
  int kk;
  int tile_num = 1024 / 128;
  int index;
  int load_flag;
  int compute_flag;
  int m;
  int n;
  int counter = 0;
  for (i = 0; i < 1024; i += 128) 
    for (j = 0; j < 1024; j += 128) {
      for (ii = 0; ii < 128; ii++) {
        for (jj = 0; jj < 128; jj++) {
          
#pragma HLS PIPELINE II = 1
          double fpTmp1 = (double )0;
          RANGE(local_prod[ii][jj / 8],jj % 8 * 64 + 63,jj % 8 * 64) =  *((long *)(&fpTmp1));
        }
      }
      for (index = 0; index < tile_num + 1; index++) {
        if (counter == 0) {
          load((index < tile_num),i,j,index * 128,local_ma_ping,local_mb_ping,ma,mb);
          compute((index > 0),local_ma_pong,local_mb_pong,local_prod);
        }
         else {
          load((index < tile_num),i,j,index * 128,local_ma_pong,local_mb_pong,ma,mb);
          compute((index > 0),local_ma_ping,local_mb_ping,local_prod);
        }
        counter = counter + 1;
        if (counter == 2) 
          counter = 0;
      }
      for (ii = 0; ii < 128; ii++) {
        for (jj = 0; jj < 128; jj++) {
          
#pragma HLS PIPELINE II = 1
          long uintTmp0 = RANGE(local_prod[ii][jj / 8],jj % 8 * 64 + 63,jj % 8 * 64);
          double castTmp0 =  *((double *)(&uintTmp0));
          double fpTmp2 = castTmp0;
          RANGE(prod[((i + ii) * 1024 + (j + jj)) / 8],((i + ii) * 1024 + (j + jj)) % 8 * 64 + 63,((i + ii) * 1024 + (j + jj)) % 8 * 64) =  *((long *)(&fpTmp2));
        }
      }
    }
  return ;
}
}
