#include "ap_int.h"
#define RANGE(var, h, l) (var).range(h, l)
#include "viterbi.h"
#include <string.h>
#include <inttypes.h>
#define JOBS_PER_BATCH 256
#define UNROLL_FACTOR 128
#define JOBS_PER_PE ((JOBS_PER_BATCH)/(UNROLL_FACTOR))

extern "C" {
int viterbi(ap_uint<128> obs[8UL],ap_uint<128> init[2UL],ap_uint<128> transition[7UL],ap_uint<128> emission[7UL],ap_uint<128> path[8UL])
{
  ap_uint<128> llike[128UL][2UL];
  uint32_t t;
  uint8_t prev;
  uint8_t curr;
  float min_p;
  float p;
  uint8_t min_s;
  uint8_t s;
// All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) )
// Initialize with first observation and initial probabilities
  L_init:
  for (s = 0; s < 5; s++) {
    
#pragma HLS PIPELINE
    uint8_t tmp = RANGE(obs[0 / 16],0 % 16 * 8 + 7,0 % 16 * 8);
    int uintTmp0 = RANGE(init[s / 4],s % 4 * 32 + 31,s % 4 * 32);
    float castTmp0 =  *((float *)(&uintTmp0));
    int uintTmp1 = RANGE(emission[(((int )s) * 5 + ((int )tmp)) / 4],(((int )s) * 5 + ((int )tmp)) % 4 * 32 + 31,(((int )s) * 5 + ((int )tmp)) % 4 * 32);
    float castTmp1 =  *((float *)(&uintTmp1));
    float fpTmp9 = castTmp0 + castTmp1;
    RANGE(llike[0][s / 4],s % 4 * 32 + 31,s % 4 * 32) =  *((int *)(&fpTmp9));
  }
// Iteratively compute the probabilities over time
  L_timestep:
  for (t = 1; t < 128; t++) {
    L_curr_state:
    for (curr = 0; curr < 5; curr++) {
// Compute likelihood HMM is in current state and where it came from.
      L_prev_state:
      for (prev = 0; prev < 5; prev++) {
        
#pragma HLS PIPELINE
        uint8_t tmp = RANGE(obs[t / 16],t % 16 * 8 + 7,t % 16 * 8);
        int uintTmp2 = RANGE(llike[t - ((unsigned int )1)][prev / 4],prev % 4 * 32 + 31,prev % 4 * 32);
        float castTmp2 =  *((float *)(&uintTmp2));
        int uintTmp3 = RANGE(transition[(((int )prev) * 5 + ((int )curr)) / 4],(((int )prev) * 5 + ((int )curr)) % 4 * 32 + 31,(((int )prev) * 5 + ((int )curr)) % 4 * 32);
        float castTmp3 =  *((float *)(&uintTmp3));
        int uintTmp4 = RANGE(emission[(((int )curr) * 5 + ((int )tmp)) / 4],(((int )curr) * 5 + ((int )tmp)) % 4 * 32 + 31,(((int )curr) * 5 + ((int )tmp)) % 4 * 32);
        float castTmp4 =  *((float *)(&uintTmp4));
        p = castTmp2 + castTmp3 + castTmp4;
        if (!prev || p < min_p) {
          min_p = p;
        }
        float fpTmp10 = min_p;
        RANGE(llike[t][curr / 4],curr % 4 * 32 + 31,curr % 4 * 32) =  *((int *)(&fpTmp10));
      }
    }
  }
// Identify end state
  min_s = 0;
  int uintTmp5 = RANGE(llike[128 - 1][min_s / 4],min_s % 4 * 32 + 31,min_s % 4 * 32);
  float castTmp5 =  *((float *)(&uintTmp5));
  min_p = castTmp5;
  L_end:
  for (s = 1; s < 5; s++) {
    
#pragma HLS PIPELINE
    int uintTmp6 = RANGE(llike[128 - 1][s / 4],s % 4 * 32 + 31,s % 4 * 32);
    float castTmp6 =  *((float *)(&uintTmp6));
    p = castTmp6;
    if (p < min_p) {
      min_p = p;
      min_s = s;
    }
  }
  RANGE(path[(128 - 1) / 16],(128 - 1) % 16 * 8 + 7,(128 - 1) % 16 * 8) = min_s;
// Backtrack to recover full path
  L_backtrack:

  for (t = (128 - 2); t >= 0; t--) {
    L_state:
/*
    for (s = 0; s < 5; s++) {
      
#pragma HLS PIPELINE II=2
      uint8_t tmp = RANGE(path[(t + ((unsigned int )1)) / 16],(t + ((unsigned int )1)) % 16 * 8 + 7,(t + ((unsigned int )1)) % 16 * 8);
      int uintTmp7 = RANGE(llike[t][s / 4],s % 4 * 32 + 31,s % 4 * 32);
      float castTmp7 =  *((float *)(&uintTmp7));
      int uintTmp8 = RANGE(transition[(((int )s) * 5 + ((int )tmp)) / 4],(((int )s) * 5 + ((int )tmp)) % 4 * 32 + 31,(((int )s) * 5 + ((int )tmp)) % 4 * 32);
      float castTmp8 =  *((float *)(&uintTmp8));
      p = castTmp7 + castTmp8;
      if (!s || p < min_p) {
        min_p = p;
        min_s = s;
      }
    }
*/
		printf("%d %d %d\n", t / 16, t % 16 * 8 + 7, t % 16 * 8);
    RANGE(path[t / 16],t % 16 * 8 + 7,t % 16 * 8) = min_s;
  }

  return 0;
}

void viterbi_tiling(ap_uint<128> *obs,ap_uint<128> *init,ap_uint<128> *transition,ap_uint<128> *emission,ap_uint<128> *path)
{
  for (int j = 0; j < 256 / 128; j++) {
    viterbi(obs + j * 128 / 16,init + j * 5 / 4,transition + j * (5 * 5) / 4,emission + j * (5 * 5) / 4,path + j * 128 / 16);
  }
}

void buffer_load(int flag,ap_uint<128> *global_buf_A,ap_uint<128> part_buf_A[128UL][16UL],ap_uint<128> *global_buf_B,ap_uint<128> part_buf_B[128UL][3UL],ap_uint<128> *global_buf_C,ap_uint<128> part_buf_C[128UL][13UL],ap_uint<128> *global_buf_D,ap_uint<128> part_buf_D[128UL][13UL])
{
  
#pragma HLS INLINE off
  int i;
  if (flag) {
    for (i = 0; i < 128; i++) {
      memcpy(part_buf_A[i],(global_buf_A + i * (128 * (256 / 128)) / 16),sizeof(uint8_t ) * 128 * (256 / 128));
    }
    for (i = 0; i < 128; i++) {
      memcpy(part_buf_B[i],(global_buf_B + i * (5 * (256 / 128)) / 4),sizeof(float ) * 5 * (256 / 128));
    }
    for (i = 0; i < 128; i++) {
      memcpy(part_buf_C[i],(global_buf_C + i * (5 * 5 * (256 / 128)) / 4),sizeof(float ) * (5 * 5) * (256 / 128));
    }
    for (i = 0; i < 128; i++) {
      memcpy(part_buf_D[i],(global_buf_D + i * (5 * 5 * (256 / 128)) / 4),sizeof(float ) * (5 * 5) * (256 / 128));
    }
  }
  return ;
}

void buffer_store(int flag,ap_uint<128> *global_buf_A,ap_uint<128> part_buf_A[128UL][16UL])
{
  
#pragma HLS INLINE off
  if (flag) {
    for (int i = 0; i < 128; i++) {
      memcpy((global_buf_A + i * (128 * (256 / 128)) / 16),part_buf_A[i],sizeof(uint8_t ) * 128 * (256 / 128));
    }
  }
  return ;
}

void buffer_compute(int flag,ap_uint<128> local_obs[128UL][16UL],ap_uint<128> local_init[128UL][3UL],ap_uint<128> local_transition[128UL][13UL],ap_uint<128> local_emission[128UL][13UL],ap_uint<128> local_path[128UL][16UL])
{
  
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j = 0; j < 128; j++) {
      
#pragma HLS UNROLL
      viterbi_tiling(local_obs[j],local_init[j],local_transition[j],local_emission[j],local_path[j]);
    }
  }
  return ;
}

void workload(ap_uint<128> *obs,ap_uint<128> *init,ap_uint<128> *transition,ap_uint<128> *emission,ap_uint<128> *path,int num_jobs)
{
  
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
  ap_uint<128> local_obs_x[128UL][16UL];
  
#pragma HLS ARRAY_PARTITION variable=local_obs_x complete dim=1
  ap_uint<128> local_obs_y[128UL][16UL];
  
#pragma HLS ARRAY_PARTITION variable=local_obs_y complete dim=1
  ap_uint<128> local_init_x[128UL][3UL];
  
#pragma HLS ARRAY_PARTITION variable=local_init_x complete dim=1
  ap_uint<128> local_init_y[128UL][3UL];
  
#pragma HLS ARRAY_PARTITION variable=local_init_y complete dim=1
  ap_uint<128> local_transition_x[128UL][13UL];
  
#pragma HLS ARRAY_PARTITION variable=local_transition_x complete dim=1
  ap_uint<128> local_transition_y[128UL][13UL];
  
#pragma HLS ARRAY_PARTITION variable=local_transition_y complete dim=1
  ap_uint<128> local_emission_x[128UL][13UL];
  
#pragma HLS ARRAY_PARTITION variable=local_emission_x complete dim=1
  ap_uint<128> local_emission_y[128UL][13UL];
  
#pragma HLS ARRAY_PARTITION variable=local_emission_y complete dim=1
  ap_uint<128> local_path_x[128UL][16UL];
  
#pragma HLS ARRAY_PARTITION variable=local_path_x complete dim=1
  ap_uint<128> local_path_y[128UL][16UL];
  
#pragma HLS ARRAY_PARTITION variable=local_path_y complete dim=1
  int num_batches = num_jobs / 256;
  int i;
  for (i = 0; i < num_batches + 2; i++) {
    int load_flag = (i >= 0 && i < num_batches);
    int compute_flag = (i >= 1 && i < num_batches + 1);
    int store_flag = (i >= 2 && i < num_batches + 2);
    if (i % 2 == 0) {
      buffer_load(load_flag,obs + i * 256 * 128 / 16,local_obs_x,init + i * 256 * 5 / 4,local_init_x,transition + i * 256 * (5 * 5) / 4,local_transition_x,emission + i * 256 * (5 * 5) / 4,local_emission_x);
      buffer_compute(compute_flag,local_obs_y,local_init_y,local_transition_y,local_emission_y,local_path_y);
      buffer_store(store_flag,path + (i - 2) * 256 / 16,local_path_x);
    }
     else {
      buffer_load(load_flag,obs + i * 256 * 128 / 16,local_obs_y,init + i * 256 * 5 / 4,local_init_y,transition + i * 256 * (5 * 5) / 4,local_transition_y,emission + i * 256 * (5 * 5) / 4,local_emission_y);
      buffer_compute(compute_flag,local_obs_x,local_init_x,local_transition_x,local_emission_x,local_path_x);
      buffer_store(store_flag,path + (i - 2) * 256 / 16,local_path_y);
    }
  }
  return ;
}
}
