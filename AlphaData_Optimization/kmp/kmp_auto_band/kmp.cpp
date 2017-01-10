#include "ap_int.h"
#define RANGE(var, h, l) (var).range(h, l)
/*
Implementation based on http://www-igm.univ-mlv.fr/~lecroq/string/node8.html
*/
/*
void CPF(char pattern[PATTERN_SIZE], int32_t kmpNext[PATTERN_SIZE]) {
    int32_t k, q;
    k = 0;
    kmpNext[0] = 0;
    c1 : for(q = 1; q < PATTERN_SIZE; q++){
        c2 : while(k > 0 && pattern[k] != pattern[q]){
            k = kmpNext[q];
        }
        if(pattern[k] == pattern[q]){
            k++;
        }
        kmpNext[q] = k;
    }
}
int kmp(char pattern[PATTERN_SIZE], char input[STRING_SIZE], int32_t kmpNext[PATTERN_SIZE], int32_t n_matches[1]) {
    int32_t i, q;
    n_matches[0] = 0;
    CPF(pattern, kmpNext);
    q = 0;
    k1 : for(i = 0; i < STRING_SIZE; i++){
        k2 : while (q > 0 && pattern[q] != input[i]){
            q = kmpNext[q];
        }
        if (pattern[q] == input[i]){
            q++;
        }
        if (q >= PATTERN_SIZE){
            n_matches[0]++;
            q = kmpNext[q - 1];
        }
    }
    return 0;
}
*/
#include "kmp.h"
#include <string.h>
extern "C" {

void kmp(ap_uint<512> pattern[1UL],ap_uint<512> input[64UL],ap_uint<512> n_matches[1UL])
{
  
#pragma HLS inline off
  ap_uint<512> input_local[1UL];
  
#pragma HLS ARRAY_PARTITION variable=input_local complete dim=1
  ap_uint<512> pattern_local[1UL];
  
#pragma HLS ARRAY_PARTITION variable=pattern_local complete dim=1
//     bool is_match[PATTERN_SIZE];
// #pragma HLS ARRAY_PARTITION variable=is_match complete dim=1
  int i;
  int j;
  for (i = 0; i < 4; i++) {
    
#pragma HLS UNROLL
    RANGE(pattern_local[i / 64],i % 64 * 8 + 7,i % 64 * 8) = RANGE(pattern[i / 64],i % 64 * 8 + 7,i % 64 * 8);
    RANGE(input_local[i / 64],i % 64 * 8 + 7,i % 64 * 8) = RANGE(input[i / 64],i % 64 * 8 + 7,i % 64 * 8);
  }
  bool is_match;
  for (i = 0; i < 256 * 1024 / 64 - 4 + 1; i++) {
    
#pragma HLS PIPELINE
    is_match = true;
    for (j = 0; j < 4; j++) {
      
#pragma HLS UNROLL
      is_match = is_match && (RANGE(pattern_local[j / 64],j % 64 * 8 + 7,j % 64 * 8)) == (RANGE(input_local[j / 64],j % 64 * 8 + 7,j % 64 * 8));
    }
    if (is_match) 
      RANGE(n_matches[0 / 16],0 % 16 * 32 + 31,0 % 16 * 32) = RANGE(n_matches[0 / 16],0 % 16 * 32 + 31,0 % 16 * 32) + 1;
    for (j = 0; j < 4 - 1; j++) {
      RANGE(input_local[j / 64],j % 64 * 8 + 7,j % 64 * 8) = RANGE(input_local[(j + 1) / 64],(j + 1) % 64 * 8 + 7,(j + 1) % 64 * 8);
    }
    RANGE(input_local[(4 - 1) / 64],(4 - 1) % 64 * 8 + 7,(4 - 1) % 64 * 8) = RANGE(input[(i + 4) / 64],(i + 4) % 64 * 8 + 7,(i + 4) % 64 * 8);
  }
}

void buffer_load(bool flag,ap_uint<512> local_buf[64UL][64UL],ap_uint<512> *global_buf)
{
  
#pragma HLS inline off
  int j;
  if (flag) {
    for (j = 0; j < 64; j++) {
      memcpy(((void *)local_buf[j]),((const void *)(global_buf + j * (256 * 1024) / 64 / 64)),sizeof(char ) * (256 * 1024 / 64));
    }
  }
}

void buffer_compute(bool flag,ap_uint<512> local_buf[64UL][64UL],ap_uint<512> pattern_buf[64UL][1UL],ap_uint<512> n_matches_buf[4UL])
{
  
#pragma HLS inline off
  int j;
  if (flag) {
    for (j = 0; j < 64; j++) {
      
#pragma HLS UNROLL
      kmp(pattern_buf[j],local_buf[j],n_matches_buf + j / 16);
    }
  }
}

void workload(ap_uint<512> pattern[1UL],ap_uint<512> input[2097152UL],ap_uint<512> n_matches[1UL])
{
  
#pragma HLS INTERFACE m_axi port=pattern offset=slave bundle=gmem
  
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
  
#pragma HLS INTERFACE m_axi port=n_matches offset=slave bundle=gmem1
  
#pragma HLS INTERFACE s_axilite port=pattern bundle=control
  
#pragma HLS INTERFACE s_axilite port=input bundle=control
  
#pragma HLS INTERFACE s_axilite port=n_matches bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  ap_uint<512> pattern_buf[64UL][1UL];
  
#pragma HLS ARRAY_PARTITION variable=pattern_buf complete dim=1
  ap_uint<512> input_buf_x[64UL][64UL];
  
#pragma HLS ARRAY_PARTITION variable=input_buf_x complete dim=1
  ap_uint<512> input_buf_y[64UL][64UL];
  
#pragma HLS ARRAY_PARTITION variable=input_buf_y complete dim=1
  ap_uint<512> n_matches_buf[4UL];
  
#pragma HLS ARRAY_PARTITION variable=n_matches_buf complete dim=1
  int i;
  int j;
  for (i = 0; i < 64; i++) {
    
#pragma HLS UNROLL
    RANGE(n_matches_buf[i / 16],i % 16 * 32 + 31,i % 16 * 32) = 0;
  }
  memcpy(pattern_buf[0],pattern,sizeof(char ) * 4);
  for (j = 0; j < 4; j++) {
    
#pragma HLS PIPELINE
    for (i = 1; i < 64; i++) {
      
#pragma HLS UNROLL
      RANGE(pattern_buf[i][j / 64],j % 64 * 8 + 7,j % 64 * 8) = RANGE(pattern_buf[0][j / 64],j % 64 * 8 + 7,j % 64 * 8);
    }
  }
  for (i = 0; i < 128 * 1024 * 1024 / (256 * 1024) + 1; i++) {
    if (i % 2 == 0) {
      buffer_load(i < 128 * 1024 * 1024 / (256 * 1024),input_buf_x,input + i * (256 * 1024) / 64);
      buffer_compute(i > 0,input_buf_y,pattern_buf,n_matches_buf);
    }
     else {
      buffer_load(i < 128 * 1024 * 1024 / (256 * 1024),input_buf_y,input + i * (256 * 1024) / 64);
      buffer_compute(i > 0,input_buf_x,pattern_buf,n_matches_buf);
    }
  }
  int final_res = 0;
  for (i = 0; i < 64; i++) {
    
#pragma HLS UNROLL
    final_res = final_res + RANGE(n_matches_buf[i / 16],i % 16 * 32 + 31,i % 16 * 32);
  }
  RANGE(n_matches[0 / 16],0 % 16 * 32 + 31,0 % 16 * 32) = final_res;
  return ;
}
}
