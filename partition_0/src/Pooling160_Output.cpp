#include "Pooling160_Output.hpp"

void Pooling160_Output_sliding_window(
    stream_t(Pooling160_Output_data_t) &in,
    stream_t(Pooling160_Output_data_t) out[POOLING160_OUTPUT_KERNEL_SIZE_X][POOLING160_OUTPUT_KERNEL_SIZE_Y]
) {
#pragma HLS INLINE OFF

    sliding_window<
        POOLING160_OUTPUT_SLIDING_WINDOW_BATCH_SIZE,
        POOLING160_OUTPUT_SLIDING_WINDOW_ROWS,
        POOLING160_OUTPUT_SLIDING_WINDOW_COLS,
        POOLING160_OUTPUT_SLIDING_WINDOW_CHANNELS,
        POOLING160_OUTPUT_SLIDING_WINDOW_PAD_TOP,
        POOLING160_OUTPUT_SLIDING_WINDOW_PAD_RIGHT,
        POOLING160_OUTPUT_SLIDING_WINDOW_PAD_BOTTOM,
        POOLING160_OUTPUT_SLIDING_WINDOW_PAD_LEFT,
        POOLING160_OUTPUT_SLIDING_WINDOW_STRIDE_X,
        POOLING160_OUTPUT_SLIDING_WINDOW_STRIDE_Y,
        POOLING160_OUTPUT_SLIDING_WINDOW_KERNEL_SIZE_X,
        POOLING160_OUTPUT_SLIDING_WINDOW_KERNEL_SIZE_Y,
        Pooling160_Output_data_t
    >(in,out);

}

void Pooling160_Output_pool(
    stream_t(Pooling160_Output_data_t) in[POOLING160_OUTPUT_KERNEL_SIZE_X][POOLING160_OUTPUT_KERNEL_SIZE_Y],
    stream_t(Pooling160_Output_data_t) &out
) {
#pragma HLS INLINE OFF

    pool<
        POOLING160_OUTPUT_POOL_BATCH_SIZE,
        POOLING160_OUTPUT_POOL_ROWS,
        POOLING160_OUTPUT_POOL_COLS,
        POOLING160_OUTPUT_POOL_CHANNELS,
        POOLING160_OUTPUT_POOL_KERNEL_SIZE_X,
        POOLING160_OUTPUT_POOL_KERNEL_SIZE_Y,
        Pooling160_Output_data_t
    >(in,out);

}

void Pooling160_Output(
    stream_t(Pooling160_Output_data_t) in[POOLING160_OUTPUT_COARSE],
    stream_t(Pooling160_Output_data_t) out[POOLING160_OUTPUT_COARSE],
    int mode
)
{

#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

    stream_t(Pooling160_Output_data_t) sw_out[POOLING160_OUTPUT_COARSE][POOLING160_OUTPUT_KERNEL_SIZE_X][POOLING160_OUTPUT_KERNEL_SIZE_Y]; //sliding window output

#pragma HLS STREAM variable=sw_out
#pragma HLS ARRAY_PARTITION variable=sw_out complete dim=0

    for(unsigned int coarse_index=0; coarse_index<POOLING160_OUTPUT_COARSE; coarse_index++)
    {
#pragma HLS UNROLL
        Pooling160_Output_sliding_window(in[coarse_index], sw_out[coarse_index]);
        Pooling160_Output_pool(sw_out[coarse_index], out[coarse_index]);
    }
}

