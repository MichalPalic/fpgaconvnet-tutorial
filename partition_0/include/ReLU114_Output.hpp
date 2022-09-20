#ifndef RELU114_OUTPUT_HPP_
#define RELU114_OUTPUT_HPP_

#include "relu.hpp"

#define name        ReLU114_Output
#define NAME        RELU114_OUTPUT
#define RELU114_OUTPUT_ID   0

#define RELU114_OUTPUT_BATCH_SIZE   1
#define RELU114_OUTPUT_ROWS         14
#define RELU114_OUTPUT_COLS         14
#define RELU114_OUTPUT_CHANNELS     16
#define RELU114_OUTPUT_COARSE       4

#define RELU114_OUTPUT_COARSE_IN    RELU114_OUTPUT_COARSE
#define RELU114_OUTPUT_COARSE_OUT   RELU114_OUTPUT_COARSE

#define RELU114_OUTPUT_ROWS_OUT     14
#define RELU114_OUTPUT_COLS_OUT     14
#define RELU114_OUTPUT_CHANNELS_OUT 16

#define RELU114_OUTPUT_RELU_BATCH_SIZE   1
#define RELU114_OUTPUT_RELU_ROWS         14
#define RELU114_OUTPUT_RELU_COLS         14
#define RELU114_OUTPUT_RELU_CHANNELS     4

typedef ap_fixed<16,8,AP_RND> ReLU114_Output_data_t;
typedef ReLU114_Output_data_t ReLU114_Output_input_t;
typedef ReLU114_Output_data_t ReLU114_Output_output_t;

/**
 * FUNCTION DEFINITION
 */

void ReLU114_Output(
    stream_t(ReLU114_Output_data_t) in[RELU114_OUTPUT_COARSE],
    stream_t(ReLU114_Output_data_t) out[RELU114_OUTPUT_COARSE],
    int mode
);

#undef name
#undef NAME
#endif
