#include "ReLU114_Output.hpp"

void ReLU114_Output_relu(
    stream_t(ReLU114_Output_data_t) &in,
    stream_t(ReLU114_Output_data_t) &out
) {

#pragma HLS INLINE OFF

    relu<
        RELU114_OUTPUT_RELU_BATCH_SIZE,
        RELU114_OUTPUT_RELU_ROWS,
        RELU114_OUTPUT_RELU_COLS,
        RELU114_OUTPUT_RELU_CHANNELS,
        ReLU114_Output_data_t
    >(in,out);

}


void ReLU114_Output(
    stream_t(ReLU114_Output_data_t) in[RELU114_OUTPUT_COARSE],
    stream_t(ReLU114_Output_data_t) out[RELU114_OUTPUT_COARSE],
    int mode
)
{

#pragma HLS INLINE OFF

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

#pragma HLS DATAFLOW

    for(unsigned int coarse_index=0; coarse_index<RELU114_OUTPUT_COARSE; coarse_index++)
    {
#pragma HLS unroll
        ReLU114_Output_relu(in[coarse_index], out[coarse_index]);
    }
}

