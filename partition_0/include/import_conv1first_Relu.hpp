#ifndef IMPORT_CONV1FIRST_RELU_HPP_
#define IMPORT_CONV1FIRST_RELU_HPP_

#include "relu.hpp"

#define name        import_conv1first_Relu
#define NAME        IMPORT_CONV1FIRST_RELU
#define IMPORT_CONV1FIRST_RELU_ID   0

#define IMPORT_CONV1FIRST_RELU_BATCH_SIZE   1
#define IMPORT_CONV1FIRST_RELU_ROWS         28
#define IMPORT_CONV1FIRST_RELU_COLS         28
#define IMPORT_CONV1FIRST_RELU_CHANNELS     32
#define IMPORT_CONV1FIRST_RELU_COARSE       2

#define IMPORT_CONV1FIRST_RELU_COARSE_IN    IMPORT_CONV1FIRST_RELU_COARSE
#define IMPORT_CONV1FIRST_RELU_COARSE_OUT   IMPORT_CONV1FIRST_RELU_COARSE

#define IMPORT_CONV1FIRST_RELU_ROWS_OUT     28
#define IMPORT_CONV1FIRST_RELU_COLS_OUT     28
#define IMPORT_CONV1FIRST_RELU_CHANNELS_OUT 32

#define IMPORT_CONV1FIRST_RELU_RELU_BATCH_SIZE   1
#define IMPORT_CONV1FIRST_RELU_RELU_ROWS         28
#define IMPORT_CONV1FIRST_RELU_RELU_COLS         28
#define IMPORT_CONV1FIRST_RELU_RELU_CHANNELS     16

typedef ap_fixed<16,8,AP_RND> import_conv1first_Relu_data_t;
typedef import_conv1first_Relu_data_t import_conv1first_Relu_input_t;
typedef import_conv1first_Relu_data_t import_conv1first_Relu_output_t;

/**
 * FUNCTION DEFINITION
 */

void import_conv1first_Relu(
    stream_t(import_conv1first_Relu_data_t) in[IMPORT_CONV1FIRST_RELU_COARSE],
    stream_t(import_conv1first_Relu_data_t) out[IMPORT_CONV1FIRST_RELU_COARSE],
    int mode
);

#undef name
#undef NAME
#endif
