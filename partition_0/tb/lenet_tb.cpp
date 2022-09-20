#include "lenet_top.hpp"
#include "common_tb.hpp"

int main()
{
    int err = 0;

    static mem_int test_in[LENET_PORTS_IN][LENET_SIZE_IN] = {0};

    // load input
    printf("LOADING INPUT DATA \n");
    load_net_data<
        LENET_PORTS_IN,
        LENET_BATCH_SIZE,
        LENET_ROWS_IN,
        LENET_COLS_IN,
        LENET_CHANNELS_IN,
        LENET_STREAMS_IN
    >("import_conv1first_Conv2D_0.dat",test_in);

    for( int wr_index=0;wr_index<LENET_WEIGHTS_RELOADING_FACTOR;wr_index++) {

        static mem_int test_out[LENET_PORTS_OUT][LENET_SIZE_OUT]          = {0};
        static mem_int test_out_valid[LENET_PORTS_OUT][LENET_SIZE_OUT]    = {0};

#if LENET_WEIGHTS_RELOADING_FLAG
        static mem_int weights[LENET_PORTS_WR][LENET_SIZE_WR] = {0};
#endif

        // load weights
        load_net_weights<
            LENET_PORTS_WR,
            LENET_SIZE_WR,
            LENET_WEIGHTS_RELOADING_FACTOR
        >("import_conv4last_BiasAdd_weights_0.dat", weights, wr_index);

        // load valid output
        load_net_data<
            LENET_PORTS_OUT,
            LENET_BATCH_SIZE,
            LENET_ROWS_OUT,
            LENET_COLS_OUT,
            LENET_CHANNELS_OUT,
            LENET_STREAMS_OUT,
            LENET_WEIGHTS_RELOADING_FACTOR
        >("import_conv4last_BiasAdd_0.dat", test_out_valid, wr_index);

        printf("RUNNING NETWORK \n");

        // perform weights reloading
        if( wr_index > 0 ) {
            fpgaconvnet_ip(1,wr_index,weights,test_in,test_out);
        }

        // run the network
        fpgaconvnet_ip(0,wr_index,weights,test_in,test_out);

        // check array is correct
        for(int i=0; i<LENET_PORTS_OUT;i++) {
            printf("PORT %d\n",i);
            err += check_array_equal<LENET_SIZE_OUT, LENET_STREAMS_OUT>(test_out[i],test_out_valid[i]);
        }

    }

    printf("%s\n",(err==0) ? "\t--- PASSED ---" : "\t--- FAILED ---");
    return err;
}
