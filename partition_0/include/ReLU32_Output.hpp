#ifndef RELU32_OUTPUT_HPP_
#define RELU32_OUTPUT_HPP_

#include "relu.hpp"

#define name        ReLU32_Output
#define NAME        RELU32_OUTPUT
#define RELU32_OUTPUT_ID   0

#define RELU32_OUTPUT_BATCH_SIZE   1
#define RELU32_OUTPUT_ROWS         28
#define RELU32_OUTPUT_COLS         28
#define RELU32_OUTPUT_CHANNELS     8
#define RELU32_OUTPUT_COARSE       1

#define RELU32_OUTPUT_COARSE_IN    RELU32_OUTPUT_COARSE
#define RELU32_OUTPUT_COARSE_OUT   RELU32_OUTPUT_COARSE

#define RELU32_OUTPUT_ROWS_OUT     28
#define RELU32_OUTPUT_COLS_OUT     28
#define RELU32_OUTPUT_CHANNELS_OUT 8

#define RELU32_OUTPUT_RELU_BATCH_SIZE   1
#define RELU32_OUTPUT_RELU_ROWS         28
#define RELU32_OUTPUT_RELU_COLS         28
#define RELU32_OUTPUT_RELU_CHANNELS     8

typedef ap_fixed<16,8,AP_RND> ReLU32_Output_data_t;
typedef ReLU32_Output_data_t ReLU32_Output_input_t;
typedef ReLU32_Output_data_t ReLU32_Output_output_t;

/**
 * FUNCTION DEFINITION
 */

void ReLU32_Output(
    stream_t(ReLU32_Output_data_t) in[RELU32_OUTPUT_COARSE],
    stream_t(ReLU32_Output_data_t) out[RELU32_OUTPUT_COARSE],
    int mode
);

#undef name
#undef NAME
#endif
