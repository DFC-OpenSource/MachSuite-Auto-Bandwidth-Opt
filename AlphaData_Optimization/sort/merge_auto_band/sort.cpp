#include "ap_int.h"
#define RANGE(var, h, l) (var).range(h, l)
#include "sort.h"
#include <string.h>
extern "C" {

void merge(ap_uint<512> a[128UL],int start,int m,int stop)
{
  ap_uint<512> temp[128UL];
  int i;
  int j;
  int k;
  merge_label1:
  for (i = start; i <= m; i++) {
    
#pragma HLS PIPELINE
    RANGE(temp[i / 16],i % 16 * 32 + 31,i % 16 * 32) = RANGE(a[i / 16],i % 16 * 32 + 31,i % 16 * 32);
  }
  merge_label2:
  for (j = m + 1; j <= stop; j++) {
    
#pragma HLS PIPELINE
    RANGE(temp[(m + 1 + stop - j) / 16],(m + 1 + stop - j) % 16 * 32 + 31,(m + 1 + stop - j) % 16 * 32) = RANGE(a[j / 16],j % 16 * 32 + 31,j % 16 * 32);
  }
  i = start;
  j = stop;
  merge_label3:
  for (k = start; k <= stop; k++) {
    
#pragma HLS PIPELINE
    int32_t tmp_j = RANGE(temp[j / 16],j % 16 * 32 + 31,j % 16 * 32);
    int32_t tmp_i = RANGE(temp[i / 16],i % 16 * 32 + 31,i % 16 * 32);
    if (tmp_j < tmp_i) {
      RANGE(a[k / 16],k % 16 * 32 + 31,k % 16 * 32) = tmp_j;
      j--;
    }
     else {
      RANGE(a[k / 16],k % 16 * 32 + 31,k % 16 * 32) = tmp_i;
      i++;
    }
  }
}

void merge_reduce(int32_t a[131072],int start,int m,int stop)
{
  int32_t temp[131072];
  int i;
  int j;
  int k;
  merge_label1:
  for (i = start; i <= m; i++) {
    
#pragma HLS PIPELINE
    temp[i] = a[i];
  }
  merge_label2:
  for (j = m + 1; j <= stop; j++) {
    
#pragma HLS PIPELINE
    temp[m + 1 + stop - j] = a[j];
  }
  i = start;
  j = stop;
  merge_label3:
  for (k = start; k <= stop; k++) {
    
#pragma HLS PIPELINE
    int32_t tmp_j = temp[j];
    int32_t tmp_i = temp[i];
    if (tmp_j < tmp_i) {
      a[k] = tmp_j;
      j--;
    }
     else {
      a[k] = tmp_i;
      i++;
    }
  }
}

void ms_mergesort(ap_uint<512> a[128UL])
{
  int start;
  int stop;
  int i;
  int m;
  int from;
  int mid;
  int to;
  start = 0;
  stop = 131072 / 64;
  mergesort_label1:
  for (m = 1; m < 131072 / 64; m += m) {
    mergesort_label2:
    for (i = start; i < stop; i += m + m) {
      merge(a,i,i + m - 1,i + 2 * m - 1);
    }
  }
}

void compute(int flag,ap_uint<512> a[64UL][128UL])
{
  if (flag) {
    int i;
    int m;
    int start = 0;
    int stop = 131072;
    for (i = 0; i < 64; i++) {
      
#pragma HLS unroll
      ms_mergesort(a[i]);
    }
/* for (m = JOBS_PER_UNROLL; m < TILING_SIZE; m += m) { */
/* mergesort_label2: */
/* 	for(i = start; i < stop; i += m + m) { */
/* 		merge_reduce(a[0], i, i + m - 1, i + 2 * m - 1); */
/* 	} */
/* } */
  }
}

void load(int flag,ap_uint<512> *local_a,ap_uint<512> *a)
{
  if (flag) {
    memcpy(local_a,a,sizeof(int32_t ) * 131072);
  }
}
/*
void load(int flag, TYPE local_a[UNROLL_FACTOR][JOBS_PER_UNROLL], TYPE *a) {
	if (flag) {
		for (int i = 0; i < UNROLL_FACTOR; i++) {
			memcpy(local_a[i], a + TILING_SIZE / UNROLL_FACTOR, 
				sizeof(TYPE) * TILING_SIZE / UNROLL_FACTOR);
		}
	}
}
*/

void save(int flag,ap_uint<512> *local_a,ap_uint<512> *a)
{
  if (flag) {
    memcpy(a,local_a,sizeof(int32_t ) * 131072);
  }
}

void workload(ap_uint<512> *a)
{
  
#pragma HLS INTERFACE m_axi offset=slave port=a bundle=gmem
  
#pragma HLS INTERFACE s_axilite port=a bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  int i;
  int iterCount = 32768 / 131072;
  ap_uint<512> local_a_0[64UL][128UL];
  
#pragma HLS ARRAY_PARTITION variable=local_a_0 complete dim=1
  ap_uint<512> local_a_1[64UL][128UL];
  
#pragma HLS ARRAY_PARTITION variable=local_a_1 complete dim=1
  ap_uint<512> local_a_2[64UL][128UL];
  
#pragma HLS ARRAY_PARTITION variable=local_a_2 complete dim=1
  for (i = 0; i < iterCount + 2; i++) {
    int idx = i % 3;
    int load_flag = (i < iterCount);
    int compute_flag = (i > 0 && i < iterCount + 1);
    int save_flag = (i > 1 && i < iterCount + 2);
    switch(idx){
      case 0:
{
        load(load_flag,local_a_0[0],a + i * 131072 / 16);
        compute(compute_flag,local_a_2);
        save(save_flag,local_a_1[0],a + (i - 2) * 131072 / 16);
        break; 
      }
      case 1:
{
        load(load_flag,local_a_1[0],a + i * 131072 / 16);
        compute(compute_flag,local_a_0);
        save(save_flag,local_a_2[0],a + (i - 2) * 131072 / 16);
        break; 
      }
      case 2:
{
        load(load_flag,local_a_2[0],a + i * 131072 / 16);
        compute(compute_flag,local_a_1);
        save(save_flag,local_a_0[0],a + (i - 2) * 131072 / 16);
        break; 
      }
      default:
      break; 
    }
  }
  return ;
}
}
