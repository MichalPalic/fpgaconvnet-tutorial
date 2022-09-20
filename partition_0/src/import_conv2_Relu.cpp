#include "import_conv2_Relu.hpp"

void import_conv2_Relu_relu(
    stream_t(import_conv2_Relu_data_t) &in,
    stream_t(import_conv2_Relu_data_t) &out
) {

#pragma HLS INLINE OFF

    relu<
        IMPORT_CONV2_RELU_RELU_BATCH_SIZE,
        IMPORT_CONV2_RELU_RELU_ROWS,
        IMPORT_CONV2_RELU_RELU_COLS,
        IMPORT_CONV2_RELU_RELU_CHANNELS,
        import_conv2_Relu_data_t
    >(in,out);

}


void import_conv2_Relu(
    stream_t(import_conv2_Relu_data_t) in[IMPORT_CONV2_RELU_COARSE],
    stream_t(import_conv2_Relu_data_t) out[IMPORT_CONV2_RELU_COARSE],
    int mode
)
{

#pragma HLS INLINE OFF

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

#pragma HLS DATAFLOW

    for(unsigned int coarse_index=0; coarse_index<IMPORT_CONV2_RELU_COARSE; coarse_index++)
    {
#pragma HLS unroll
        import_conv2_Relu_relu(in[coarse_index], out[coarse_index]);
    }
}

