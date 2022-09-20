#include "Pooling66_Output.hpp"

void Pooling66_Output_sliding_window(
    stream_t(Pooling66_Output_data_t) &in,
    stream_t(Pooling66_Output_data_t) out[POOLING66_OUTPUT_KERNEL_SIZE_X][POOLING66_OUTPUT_KERNEL_SIZE_Y]
) {
#pragma HLS INLINE OFF

    sliding_window<
        POOLING66_OUTPUT_SLIDING_WINDOW_BATCH_SIZE,
        POOLING66_OUTPUT_SLIDING_WINDOW_ROWS,
        POOLING66_OUTPUT_SLIDING_WINDOW_COLS,
        POOLING66_OUTPUT_SLIDING_WINDOW_CHANNELS,
        POOLING66_OUTPUT_SLIDING_WINDOW_PAD_TOP,
        POOLING66_OUTPUT_SLIDING_WINDOW_PAD_RIGHT,
        POOLING66_OUTPUT_SLIDING_WINDOW_PAD_BOTTOM,
        POOLING66_OUTPUT_SLIDING_WINDOW_PAD_LEFT,
        POOLING66_OUTPUT_SLIDING_WINDOW_STRIDE_X,
        POOLING66_OUTPUT_SLIDING_WINDOW_STRIDE_Y,
        POOLING66_OUTPUT_SLIDING_WINDOW_KERNEL_SIZE_X,
        POOLING66_OUTPUT_SLIDING_WINDOW_KERNEL_SIZE_Y,
        Pooling66_Output_data_t
    >(in,out);

}

void Pooling66_Output_pool(
    stream_t(Pooling66_Output_data_t) in[POOLING66_OUTPUT_KERNEL_SIZE_X][POOLING66_OUTPUT_KERNEL_SIZE_Y],
    stream_t(Pooling66_Output_data_t) &out
) {
#pragma HLS INLINE OFF

    pool<
        POOLING66_OUTPUT_POOL_BATCH_SIZE,
        POOLING66_OUTPUT_POOL_ROWS,
        POOLING66_OUTPUT_POOL_COLS,
        POOLING66_OUTPUT_POOL_CHANNELS,
        POOLING66_OUTPUT_POOL_KERNEL_SIZE_X,
        POOLING66_OUTPUT_POOL_KERNEL_SIZE_Y,
        Pooling66_Output_data_t
    >(in,out);

}

void Pooling66_Output(
    stream_t(Pooling66_Output_data_t) in[POOLING66_OUTPUT_COARSE],
    stream_t(Pooling66_Output_data_t) out[POOLING66_OUTPUT_COARSE],
    int mode
)
{

#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

    stream_t(Pooling66_Output_data_t) sw_out[POOLING66_OUTPUT_COARSE][POOLING66_OUTPUT_KERNEL_SIZE_X][POOLING66_OUTPUT_KERNEL_SIZE_Y]; //sliding window output

#pragma HLS STREAM variable=sw_out
#pragma HLS ARRAY_PARTITION variable=sw_out complete dim=0

    for(unsigned int coarse_index=0; coarse_index<POOLING66_OUTPUT_COARSE; coarse_index++)
    {
#pragma HLS UNROLL
        Pooling66_Output_sliding_window(in[coarse_index], sw_out[coarse_index]);
        Pooling66_Output_pool(sw_out[coarse_index], out[coarse_index]);
    }
}

