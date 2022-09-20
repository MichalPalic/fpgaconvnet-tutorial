#ifndef LENET_TOP_HPP_
#define LENET_TOP_HPP_

#include "common.hpp"
#include "import_conv1first_Conv2D.hpp"
#include "import_conv1first_Relu.hpp"
#include "import_pool1_MaxPool.hpp"
#include "import_conv2_Conv2D.hpp"
#include "import_conv2_Relu.hpp"
#include "import_pool2_MaxPool.hpp"
#include "import_conv3_Conv2D.hpp"
#include "import_conv3_Relu.hpp"
#include "import_conv4last_BiasAdd.hpp"

#include "mem_read.hpp"
#include "mem_write.hpp"
#include "wr.hpp"

#define LENET_BATCH_SIZE   1

#define LENET_ROWS_IN      28
#define LENET_COLS_IN      28
#define LENET_CHANNELS_IN  1

#define LENET_ROWS_OUT     1
#define LENET_COLS_OUT     1
#define LENET_CHANNELS_OUT 10

#define LENET_STREAMS_IN   1
#define LENET_STREAMS_OUT  1
#define LENET_STREAMS_WR   1

#define LENET_PORTS        1
#define LENET_PORTS_IN     1  //LENET_PORTS
#define LENET_PORTS_OUT    1  //LENET_PORTS
#define LENET_PORTS_WR     1 //LENET_PORTS

#define LENET_WEIGHTS_RELOADING_FACTOR 1
#define LENET_WEIGHTS_RELOADING_LAYER  import_conv4last_BiasAdd
#define LENET_WEIGHTS_RELOADING_FLAG   1

#define LENET_SIZE_IN  LENET_BATCH_SIZE*LENET_ROWS_IN*LENET_COLS_IN*DIVIDE(LENET_CHANNELS_IN,LENET_STREAMS_IN)
#define LENET_SIZE_OUT LENET_BATCH_SIZE*LENET_ROWS_OUT*LENET_COLS_OUT*DIVIDE(LENET_CHANNELS_OUT,LENET_STREAMS_OUT)*LENET_WEIGHTS_RELOADING_FACTOR

typedef import_conv1first_Conv2D_input_t   lenet_input_t;
typedef import_conv4last_BiasAdd_output_t lenet_output_t;

#if LENET_WEIGHTS_RELOADING_FLAG
#define LENET_WR_COARSE_IN       IMPORT_CONV4LAST_BIASADD_COARSE_IN
#define LENET_WR_COARSE_OUT      IMPORT_CONV4LAST_BIASADD_COARSE_OUT
#define LENET_WR_COARSE_GROUP    IMPORT_CONV4LAST_BIASADD_COARSE_GROUP
#define LENET_WR_WEIGHTS         IMPORT_CONV4LAST_BIASADD_WEIGHTS
#define LENET_WR_KERNEL_SIZE_X   IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X
#define LENET_WR_KERNEL_SIZE_Y   IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y

#define LENET_SIZE_WR  DIVIDE(LENET_WR_WEIGHTS,LENET_STREAMS_WR)

#define LENET_WR_BATCH_SIZE    1
#define LENET_WR_ROWS_IN       1
#define LENET_WR_COLS_IN       1
#define LENET_WR_CHANNELS_IN   LENET_SIZE_WR
#define LENET_WR_PORTS_IN      LENET_PORTS_WR
#define LENET_WR_STREAMS_IN    LENET_STREAMS_WR

void reload_weights(
    int weights_reloading_index,
    volatile mem_int wr_hw[LENET_PORTS_WR][LENET_SIZE_WR],
    import_conv4last_BiasAdd_weight_t weights[LENET_WR_COARSE_IN*LENET_WR_COARSE_GROUP][LENET_WR_COARSE_OUT][DIVIDE(LENET_WR_WEIGHTS,LENET_WR_COARSE_IN*LENET_WR_COARSE_GROUP*LENET_WR_COARSE_OUT*LENET_WR_KERNEL_SIZE_X*LENET_WR_KERNEL_SIZE_Y)][LENET_WR_KERNEL_SIZE_X][LENET_WR_KERNEL_SIZE_Y]
);
#endif

void process(
    int weights_reloading_index,
    volatile mem_int in_hw[LENET_PORTS_IN][LENET_SIZE_IN],
    volatile mem_int out_hw[LENET_PORTS_OUT][LENET_SIZE_OUT]
);

void fpgaconvnet_ip(
    int mode,
    int weights_reloading_index,
#if LENET_WEIGHTS_RELOADING_FLAG
    volatile mem_int wr_hw[LENET_PORTS_WR][LENET_SIZE_WR],
#endif
    volatile mem_int in_hw[LENET_PORTS_IN][LENET_SIZE_IN],
    volatile mem_int out_hw[LENET_PORTS_OUT][LENET_SIZE_OUT]
);

#endif
