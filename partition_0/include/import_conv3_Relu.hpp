#ifndef IMPORT_CONV3_RELU_HPP_
#define IMPORT_CONV3_RELU_HPP_

#include "relu.hpp"

#define name        import_conv3_Relu
#define NAME        IMPORT_CONV3_RELU
#define IMPORT_CONV3_RELU_ID   0

#define IMPORT_CONV3_RELU_BATCH_SIZE   1
#define IMPORT_CONV3_RELU_ROWS         1
#define IMPORT_CONV3_RELU_COLS         1
#define IMPORT_CONV3_RELU_CHANNELS     1024
#define IMPORT_CONV3_RELU_COARSE       1

#define IMPORT_CONV3_RELU_COARSE_IN    IMPORT_CONV3_RELU_COARSE
#define IMPORT_CONV3_RELU_COARSE_OUT   IMPORT_CONV3_RELU_COARSE

#define IMPORT_CONV3_RELU_ROWS_OUT     1
#define IMPORT_CONV3_RELU_COLS_OUT     1
#define IMPORT_CONV3_RELU_CHANNELS_OUT 1024

#define IMPORT_CONV3_RELU_RELU_BATCH_SIZE   1
#define IMPORT_CONV3_RELU_RELU_ROWS         1
#define IMPORT_CONV3_RELU_RELU_COLS         1
#define IMPORT_CONV3_RELU_RELU_CHANNELS     1024

typedef ap_fixed<16,8,AP_RND> import_conv3_Relu_data_t;
typedef import_conv3_Relu_data_t import_conv3_Relu_input_t;
typedef import_conv3_Relu_data_t import_conv3_Relu_output_t;

/**
 * FUNCTION DEFINITION
 */

void import_conv3_Relu(
    stream_t(import_conv3_Relu_data_t) in[IMPORT_CONV3_RELU_COARSE],
    stream_t(import_conv3_Relu_data_t) out[IMPORT_CONV3_RELU_COARSE],
    int mode
);

#undef name
#undef NAME
#endif
