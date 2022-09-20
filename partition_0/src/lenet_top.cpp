#include "lenet_top.hpp"


const static import_conv1first_Conv2D_weight_t import_conv1first_Conv2D_weights[IMPORT_CONV1FIRST_CONV2D_COARSE_IN*IMPORT_CONV1FIRST_CONV2D_COARSE_GROUP][IMPORT_CONV1FIRST_CONV2D_COARSE_OUT][DIVIDE(IMPORT_CONV1FIRST_CONV2D_WEIGHTS,IMPORT_CONV1FIRST_CONV2D_COARSE_IN*IMPORT_CONV1FIRST_CONV2D_COARSE_GROUP*IMPORT_CONV1FIRST_CONV2D_COARSE_OUT*IMPORT_CONV1FIRST_CONV2D_KERNEL_SIZE_X*IMPORT_CONV1FIRST_CONV2D_KERNEL_SIZE_Y)][IMPORT_CONV1FIRST_CONV2D_KERNEL_SIZE_X][IMPORT_CONV1FIRST_CONV2D_KERNEL_SIZE_Y] = {
#include "import_conv1first_Conv2D_weights_0.csv"
};
        


const static import_conv2_Conv2D_weight_t import_conv2_Conv2D_weights[IMPORT_CONV2_CONV2D_COARSE_IN*IMPORT_CONV2_CONV2D_COARSE_GROUP][IMPORT_CONV2_CONV2D_COARSE_OUT][DIVIDE(IMPORT_CONV2_CONV2D_WEIGHTS,IMPORT_CONV2_CONV2D_COARSE_IN*IMPORT_CONV2_CONV2D_COARSE_GROUP*IMPORT_CONV2_CONV2D_COARSE_OUT*IMPORT_CONV2_CONV2D_KERNEL_SIZE_X*IMPORT_CONV2_CONV2D_KERNEL_SIZE_Y)][IMPORT_CONV2_CONV2D_KERNEL_SIZE_X][IMPORT_CONV2_CONV2D_KERNEL_SIZE_Y] = {
#include "import_conv2_Conv2D_weights_0.csv"
};
        


const static import_conv3_Conv2D_weight_t import_conv3_Conv2D_weights[IMPORT_CONV3_CONV2D_COARSE_IN*IMPORT_CONV3_CONV2D_COARSE_GROUP][IMPORT_CONV3_CONV2D_COARSE_OUT][DIVIDE(IMPORT_CONV3_CONV2D_WEIGHTS,IMPORT_CONV3_CONV2D_COARSE_IN*IMPORT_CONV3_CONV2D_COARSE_GROUP*IMPORT_CONV3_CONV2D_COARSE_OUT*IMPORT_CONV3_CONV2D_KERNEL_SIZE_X*IMPORT_CONV3_CONV2D_KERNEL_SIZE_Y)][IMPORT_CONV3_CONV2D_KERNEL_SIZE_X][IMPORT_CONV3_CONV2D_KERNEL_SIZE_Y] = {
#include "import_conv3_Conv2D_weights_0.csv"
};
        


static import_conv4last_BiasAdd_weight_t import_conv4last_BiasAdd_weights[IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP][IMPORT_CONV4LAST_BIASADD_COARSE_OUT][DIVIDE(IMPORT_CONV4LAST_BIASADD_WEIGHTS,IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP*IMPORT_CONV4LAST_BIASADD_COARSE_OUT*IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X*IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y)][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y] = {
#include "import_conv4last_BiasAdd_weights_0.csv"
};
        


const static import_conv1first_Conv2D_biases_t import_conv1first_Conv2D_biases[IMPORT_CONV1FIRST_CONV2D_COARSE_OUT][DIVIDE(IMPORT_CONV1FIRST_CONV2D_FILTERS,IMPORT_CONV1FIRST_CONV2D_COARSE_OUT)] = {
#include "import_conv1first_Conv2D_biases.csv"
};
        


const static import_conv2_Conv2D_biases_t import_conv2_Conv2D_biases[IMPORT_CONV2_CONV2D_COARSE_OUT][DIVIDE(IMPORT_CONV2_CONV2D_FILTERS,IMPORT_CONV2_CONV2D_COARSE_OUT)] = {
#include "import_conv2_Conv2D_biases.csv"
};
        


const static import_conv3_Conv2D_biases_t import_conv3_Conv2D_biases[IMPORT_CONV3_CONV2D_COARSE_OUT][DIVIDE(IMPORT_CONV3_CONV2D_FILTERS,IMPORT_CONV3_CONV2D_COARSE_OUT)] = {
#include "import_conv3_Conv2D_biases.csv"
};
        


const static import_conv4last_BiasAdd_biases_t import_conv4last_BiasAdd_biases[IMPORT_CONV4LAST_BIASADD_COARSE_OUT][DIVIDE(IMPORT_CONV4LAST_BIASADD_FILTERS,IMPORT_CONV4LAST_BIASADD_COARSE_OUT)] = {
#include "import_conv4last_BiasAdd_biases.csv"
};
        

#if LENET_WEIGHTS_RELOADING_FLAG
void reload_weights(
    int weights_reloading_index,
    volatile mem_int wr_hw[LENET_PORTS_WR][LENET_SIZE_WR],
    import_conv4last_BiasAdd_weight_t weights[LENET_WR_COARSE_IN*LENET_WR_COARSE_GROUP][LENET_WR_COARSE_OUT][DIVIDE(LENET_WR_WEIGHTS,LENET_WR_COARSE_IN*LENET_WR_COARSE_GROUP*LENET_WR_COARSE_OUT*LENET_WR_KERNEL_SIZE_X*LENET_WR_KERNEL_SIZE_Y)][LENET_WR_KERNEL_SIZE_X][LENET_WR_KERNEL_SIZE_Y]
)
{

#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

#pragma HLS stable variable=weights

    // stream init
    stream_t(import_conv4last_BiasAdd_weight_t) wr[LENET_STREAMS_WR];
#pragma HLS STREAM variable=wr
#pragma HLS ARRAY_PARTITION variable=wr complete dim=0

    mem_read<
        LENET_WR_BATCH_SIZE,
        LENET_WR_ROWS_IN,
        LENET_WR_COLS_IN,
        LENET_WR_CHANNELS_IN,
        LENET_WR_PORTS_IN,
        LENET_WR_STREAMS_IN,
        import_conv4last_BiasAdd_weight_t
    >(wr_hw,wr);

    weights_reloading<
       LENET_WR_WEIGHTS,
       LENET_WR_COARSE_IN,
       LENET_WR_COARSE_OUT,
       LENET_WR_COARSE_GROUP,
       LENET_WR_KERNEL_SIZE_X,
       LENET_WR_KERNEL_SIZE_Y,
       import_conv4last_BiasAdd_weight_t
    >(wr[0],weights);
}
#endif

void process(
    int weights_reloading_index,
    volatile mem_int in_hw[LENET_PORTS_IN][LENET_SIZE_IN],
    volatile mem_int out_hw[LENET_PORTS_OUT][LENET_SIZE_OUT]
)
{

#pragma HLS INLINE OFF
#pragma HLS DATAFLOW


#pragma HLS ARRAY_PARTITION variable=import_conv1first_Conv2D_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=import_conv1first_Conv2D_weights complete dim=2
#pragma HLS RESOURCE variable=import_conv1first_Conv2D_weights core=ROM
#pragma HLS STABLE variable=import_conv1first_Conv2D_weights
        


#pragma HLS ARRAY_PARTITION variable=import_conv2_Conv2D_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=import_conv2_Conv2D_weights complete dim=2
#pragma HLS RESOURCE variable=import_conv2_Conv2D_weights core=ROM
#pragma HLS STABLE variable=import_conv2_Conv2D_weights
        


#pragma HLS ARRAY_PARTITION variable=import_conv3_Conv2D_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=import_conv3_Conv2D_weights complete dim=2
#pragma HLS RESOURCE variable=import_conv3_Conv2D_weights core=ROM
#pragma HLS STABLE variable=import_conv3_Conv2D_weights
        


#pragma HLS ARRAY_PARTITION variable=import_conv4last_BiasAdd_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=import_conv4last_BiasAdd_weights complete dim=2
#pragma HLS RESOURCE variable=import_conv4last_BiasAdd_weights core=RAM
#pragma HLS STABLE variable=import_conv4last_BiasAdd_weights
        

#pragma HLS ARRAY_PARTITION variable=import_conv1first_Conv2D_biases complete dim=1
#pragma HLS RESOURCE variable=import_conv1first_Conv2D_biases core=ROM
#pragma HLS STABLE variable=import_conv1first_Conv2D_biases
        


#pragma HLS ARRAY_PARTITION variable=import_conv2_Conv2D_biases complete dim=1
#pragma HLS RESOURCE variable=import_conv2_Conv2D_biases core=ROM
#pragma HLS STABLE variable=import_conv2_Conv2D_biases
        


#pragma HLS ARRAY_PARTITION variable=import_conv3_Conv2D_biases complete dim=1
#pragma HLS RESOURCE variable=import_conv3_Conv2D_biases core=ROM
#pragma HLS STABLE variable=import_conv3_Conv2D_biases
        


#pragma HLS ARRAY_PARTITION variable=import_conv4last_BiasAdd_biases complete dim=1
#pragma HLS RESOURCE variable=import_conv4last_BiasAdd_biases core=ROM
#pragma HLS STABLE variable=import_conv4last_BiasAdd_biases
        

    stream_t(import_conv1first_Conv2D_input_t) in[IMPORT_CONV1FIRST_CONV2D_COARSE_IN];
#pragma HLS STREAM variable=in
#pragma HLS ARRAY_PARTITION variable=in complete dim=0
        

    stream_t(import_conv1first_Relu_input_t) import_conv1first_Conv2D_import_conv1first_Relu[IMPORT_CONV1FIRST_RELU_COARSE_IN];
#pragma HLS STREAM variable=import_conv1first_Conv2D_import_conv1first_Relu
#pragma HLS ARRAY_PARTITION variable=import_conv1first_Conv2D_import_conv1first_Relu complete dim=0
        

    stream_t(import_pool1_MaxPool_input_t) import_conv1first_Relu_import_pool1_MaxPool[IMPORT_POOL1_MAXPOOL_COARSE_IN];
#pragma HLS STREAM variable=import_conv1first_Relu_import_pool1_MaxPool
#pragma HLS ARRAY_PARTITION variable=import_conv1first_Relu_import_pool1_MaxPool complete dim=0
        

    stream_t(import_conv2_Conv2D_input_t) import_pool1_MaxPool_import_conv2_Conv2D[IMPORT_CONV2_CONV2D_COARSE_IN];
#pragma HLS STREAM variable=import_pool1_MaxPool_import_conv2_Conv2D
#pragma HLS ARRAY_PARTITION variable=import_pool1_MaxPool_import_conv2_Conv2D complete dim=0
        

    stream_t(import_conv2_Relu_input_t) import_conv2_Conv2D_import_conv2_Relu[IMPORT_CONV2_RELU_COARSE_IN];
#pragma HLS STREAM variable=import_conv2_Conv2D_import_conv2_Relu
#pragma HLS ARRAY_PARTITION variable=import_conv2_Conv2D_import_conv2_Relu complete dim=0
        

    stream_t(import_pool2_MaxPool_input_t) import_conv2_Relu_import_pool2_MaxPool[IMPORT_POOL2_MAXPOOL_COARSE_IN];
#pragma HLS STREAM variable=import_conv2_Relu_import_pool2_MaxPool
#pragma HLS ARRAY_PARTITION variable=import_conv2_Relu_import_pool2_MaxPool complete dim=0
        

    stream_t(import_conv3_Conv2D_input_t) import_pool2_MaxPool_import_conv3_Conv2D[IMPORT_CONV3_CONV2D_COARSE_IN];
#pragma HLS STREAM variable=import_pool2_MaxPool_import_conv3_Conv2D
#pragma HLS ARRAY_PARTITION variable=import_pool2_MaxPool_import_conv3_Conv2D complete dim=0
        

    stream_t(import_conv3_Relu_input_t) import_conv3_Conv2D_import_conv3_Relu[IMPORT_CONV3_RELU_COARSE_IN];
#pragma HLS STREAM variable=import_conv3_Conv2D_import_conv3_Relu
#pragma HLS ARRAY_PARTITION variable=import_conv3_Conv2D_import_conv3_Relu complete dim=0
        

    stream_t(import_conv4last_BiasAdd_input_t) import_conv3_Relu_import_conv4last_BiasAdd[IMPORT_CONV4LAST_BIASADD_COARSE_IN];
#pragma HLS STREAM variable=import_conv3_Relu_import_conv4last_BiasAdd
#pragma HLS ARRAY_PARTITION variable=import_conv3_Relu_import_conv4last_BiasAdd complete dim=0
        

    stream_t(import_conv4last_BiasAdd_output_t) out[IMPORT_CONV4LAST_BIASADD_COARSE_OUT];
#pragma HLS STREAM variable=out
#pragma HLS ARRAY_PARTITION variable=out complete dim=0
        

    mem_read<
        LENET_BATCH_SIZE,
        LENET_ROWS_IN,
        LENET_COLS_IN,
        LENET_CHANNELS_IN,
        LENET_PORTS_IN,
        LENET_STREAMS_IN,
        lenet_input_t
    >(in_hw,in);

    int mode = 0;

    import_conv1first_Conv2D(import_conv1first_Conv2D_weights, import_conv1first_Conv2D_biases, in, import_conv1first_Conv2D_import_conv1first_Relu, mode);
    import_conv1first_Relu(import_conv1first_Conv2D_import_conv1first_Relu, import_conv1first_Relu_import_pool1_MaxPool, mode);
    import_pool1_MaxPool(import_conv1first_Relu_import_pool1_MaxPool, import_pool1_MaxPool_import_conv2_Conv2D, mode);
    import_conv2_Conv2D(import_conv2_Conv2D_weights, import_conv2_Conv2D_biases, import_pool1_MaxPool_import_conv2_Conv2D, import_conv2_Conv2D_import_conv2_Relu, mode);
    import_conv2_Relu(import_conv2_Conv2D_import_conv2_Relu, import_conv2_Relu_import_pool2_MaxPool, mode);
    import_pool2_MaxPool(import_conv2_Relu_import_pool2_MaxPool, import_pool2_MaxPool_import_conv3_Conv2D, mode);
    import_conv3_Conv2D(import_conv3_Conv2D_weights, import_conv3_Conv2D_biases, import_pool2_MaxPool_import_conv3_Conv2D, import_conv3_Conv2D_import_conv3_Relu, mode);
    import_conv3_Relu(import_conv3_Conv2D_import_conv3_Relu, import_conv3_Relu_import_conv4last_BiasAdd, mode);
    import_conv4last_BiasAdd(import_conv4last_BiasAdd_weights, import_conv4last_BiasAdd_biases, import_conv3_Relu_import_conv4last_BiasAdd, out, mode);


    mem_write<
        LENET_BATCH_SIZE,
        LENET_ROWS_OUT,
        LENET_COLS_OUT,
        LENET_CHANNELS_OUT,
        LENET_PORTS_OUT,
        LENET_STREAMS_OUT,
        LENET_WEIGHTS_RELOADING_FACTOR,
        lenet_output_t
    >(weights_reloading_index,out,out_hw);

}

void fpgaconvnet_ip(
    int mode,
    int weights_reloading_index,
#if LENET_WEIGHTS_RELOADING_FLAG
    volatile mem_int wr_hw[LENET_PORTS_WR][LENET_SIZE_WR],
#endif
    volatile mem_int in_hw[LENET_PORTS_IN][LENET_SIZE_IN],
    volatile mem_int out_hw[LENET_PORTS_OUT][LENET_SIZE_OUT]
)
{
#pragma HLS INTERFACE s_axilite port=return                     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=mode                       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=weights_reloading_index    bundle=ctrl

#if LENET_WEIGHTS_RELOADING_FLAG
#pragma HLS ARRAY_PARTITION variable=wr_hw  complete dim=1
#endif
#pragma HLS ARRAY_PARTITION variable=in_hw  complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_hw complete dim=1

#if LENET_WEIGHTS_RELOADING_FLAG
    const unsigned size_wr  = LENET_SIZE_WR ;
#endif
    const unsigned size_in  = LENET_SIZE_IN ;
    const unsigned size_out = LENET_SIZE_OUT;

#if LENET_WEIGHTS_RELOADING_FLAG
#pragma HLS INTERFACE m_axi port=wr_hw  offset=slave depth=size_wr  num_read_outstanding=1 num_write_outstanding=1 max_read_burst_length=256 max_write_burst_length=256 name=fpgaconvnet_wr  bundle=fpgaconvnet_port_wr
#endif

#pragma HLS INTERFACE m_axi port=in_hw  offset=slave depth=size_in  num_read_outstanding=1 num_write_outstanding=1 max_read_burst_length=256 max_write_burst_length=256 name=fpgaconvnet_in  bundle=fpgaconvnet_port_in

#pragma HLS INTERFACE m_axi port=out_hw offset=slave depth=size_out num_read_outstanding=1 num_write_outstanding=1 max_read_burst_length=256 max_write_burst_length=256 name=fpgaconvnet_out bundle=fpgaconvnet_port_out


    #pragma HLS DATAFLOW
    if ( mode == 0 ) {
        process(weights_reloading_index,in_hw,out_hw);
    } else if ( mode == 1 ) {
#if LENET_WEIGHTS_RELOADING_FLAG
        reload_weights(weights_reloading_index,wr_hw,import_conv4last_BiasAdd_weights);
#endif
    }

}
