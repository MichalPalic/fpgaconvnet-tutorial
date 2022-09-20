#include "import_pool1_MaxPool.hpp"

void import_pool1_MaxPool_sliding_window(
    stream_t(import_pool1_MaxPool_data_t) &in,
    stream_t(import_pool1_MaxPool_data_t) out[IMPORT_POOL1_MAXPOOL_KERNEL_SIZE_X][IMPORT_POOL1_MAXPOOL_KERNEL_SIZE_Y]
) {
#pragma HLS INLINE OFF

    sliding_window<
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_BATCH_SIZE,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_ROWS,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_COLS,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_CHANNELS,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_PAD_TOP,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_PAD_RIGHT,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_PAD_BOTTOM,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_PAD_LEFT,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_STRIDE_X,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_STRIDE_Y,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_KERNEL_SIZE_X,
        IMPORT_POOL1_MAXPOOL_SLIDING_WINDOW_KERNEL_SIZE_Y,
        import_pool1_MaxPool_data_t
    >(in,out);

}

void import_pool1_MaxPool_pool(
    stream_t(import_pool1_MaxPool_data_t) in[IMPORT_POOL1_MAXPOOL_KERNEL_SIZE_X][IMPORT_POOL1_MAXPOOL_KERNEL_SIZE_Y],
    stream_t(import_pool1_MaxPool_data_t) &out
) {
#pragma HLS INLINE OFF

    pool<
        IMPORT_POOL1_MAXPOOL_POOL_BATCH_SIZE,
        IMPORT_POOL1_MAXPOOL_POOL_ROWS,
        IMPORT_POOL1_MAXPOOL_POOL_COLS,
        IMPORT_POOL1_MAXPOOL_POOL_CHANNELS,
        IMPORT_POOL1_MAXPOOL_POOL_KERNEL_SIZE_X,
        IMPORT_POOL1_MAXPOOL_POOL_KERNEL_SIZE_Y,
        import_pool1_MaxPool_data_t
    >(in,out);

}

void import_pool1_MaxPool(
    stream_t(import_pool1_MaxPool_data_t) in[IMPORT_POOL1_MAXPOOL_COARSE],
    stream_t(import_pool1_MaxPool_data_t) out[IMPORT_POOL1_MAXPOOL_COARSE],
    int mode
)
{

#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

    stream_t(import_pool1_MaxPool_data_t) sw_out[IMPORT_POOL1_MAXPOOL_COARSE][IMPORT_POOL1_MAXPOOL_KERNEL_SIZE_X][IMPORT_POOL1_MAXPOOL_KERNEL_SIZE_Y]; //sliding window output

#pragma HLS STREAM variable=sw_out
#pragma HLS ARRAY_PARTITION variable=sw_out complete dim=0

    for(unsigned int coarse_index=0; coarse_index<IMPORT_POOL1_MAXPOOL_COARSE; coarse_index++)
    {
#pragma HLS UNROLL
        import_pool1_MaxPool_sliding_window(in[coarse_index], sw_out[coarse_index]);
        import_pool1_MaxPool_pool(sw_out[coarse_index], out[coarse_index]);
    }
}

