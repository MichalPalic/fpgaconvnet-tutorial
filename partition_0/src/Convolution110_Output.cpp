#include "Convolution110_Output.hpp"

void Convolution110_Output_sliding_window(
    stream_t(Convolution110_Output_input_t)  &in,
    stream_t(Convolution110_Output_output_t) out[CONVOLUTION110_OUTPUT_KERNEL_SIZE_X][CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y]
) {

#pragma HLS INLINE OFF

    sliding_window<
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_BATCH_SIZE,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_ROWS,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_COLS,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_CHANNELS,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_PAD_TOP,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_PAD_RIGHT,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_PAD_BOTTOM,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_PAD_LEFT,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_STRIDE_X,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_STRIDE_Y,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_KERNEL_SIZE_X,
        CONVOLUTION110_OUTPUT_SLIDING_WINDOW_KERNEL_SIZE_Y,
        Convolution110_Output_input_t
    >(in,out);

}

void Convolution110_Output_fork(
#if CONVOLUTION110_OUTPUT_KERNEL_SIZE_X == 1 && CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y == 1
    stream_t(Convolution110_Output_input_t)  &in,
    stream_t(Convolution110_Output_output_t) out[CONVOLUTION110_OUTPUT_COARSE_OUT]
#else
    stream_t(Convolution110_Output_input_t)  in[CONVOLUTION110_OUTPUT_KERNEL_SIZE_X][CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y],
    stream_t(Convolution110_Output_output_t) out[CONVOLUTION110_OUTPUT_COARSE_OUT][CONVOLUTION110_OUTPUT_KERNEL_SIZE_X][CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y]
#endif
) {

#pragma HLS INLINE OFF

    fork<
        CONVOLUTION110_OUTPUT_FORK_BATCH_SIZE,
        CONVOLUTION110_OUTPUT_FORK_ROWS,
        CONVOLUTION110_OUTPUT_FORK_COLS,
        CONVOLUTION110_OUTPUT_FORK_CHANNELS,
        CONVOLUTION110_OUTPUT_FORK_COARSE,
#if CONVOLUTION110_OUTPUT_FORK_KERNEL_SIZE_X > 1 || CONVOLUTION110_OUTPUT_FORK_KERNEL_SIZE_Y > 1
        CONVOLUTION110_OUTPUT_FORK_KERNEL_SIZE_X,
        CONVOLUTION110_OUTPUT_FORK_KERNEL_SIZE_Y,
#endif
        Convolution110_Output_input_t
    >(in,out);

}

void Convolution110_Output_conv(
    const Convolution110_Output_weight_t weights[DIVIDE(CONVOLUTION110_OUTPUT_WEIGHTS,CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP*CONVOLUTION110_OUTPUT_COARSE_OUT*CONVOLUTION110_OUTPUT_KERNEL_SIZE_X*CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y)][CONVOLUTION110_OUTPUT_KERNEL_SIZE_X][CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y],
#if CONVOLUTION110_OUTPUT_KERNEL_SIZE_X == 1 && CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y == 1
    stream_t(Convolution110_Output_input_t) &in,
#else
    stream_t(Convolution110_Output_input_t)  in[CONVOLUTION110_OUTPUT_KERNEL_SIZE_X][CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y],
#endif
    stream_t(Convolution110_Output_acc_t) &out
) {

#pragma HLS INLINE OFF

    conv<
        CONVOLUTION110_OUTPUT_CONV_BATCH_SIZE,
        CONVOLUTION110_OUTPUT_CONV_ROWS,
        CONVOLUTION110_OUTPUT_CONV_COLS,
        CONVOLUTION110_OUTPUT_CONV_CHANNELS,
        CONVOLUTION110_OUTPUT_CONV_FILTERS,
        CONVOLUTION110_OUTPUT_CONV_GROUPS,
#if (CONVOLUTION110_OUTPUT_CONV_KERNEL_SIZE_X > 1) || (CONVOLUTION110_OUTPUT_CONV_KERNEL_SIZE_Y > 1)
        CONVOLUTION110_OUTPUT_CONV_FINE,
        CONVOLUTION110_OUTPUT_CONV_KERNEL_SIZE_X,
        CONVOLUTION110_OUTPUT_CONV_KERNEL_SIZE_Y,
#endif
        Convolution110_Output_input_t,
        Convolution110_Output_weight_t,
        Convolution110_Output_acc_t
    >(in,weights,out);

}

void Convolution110_Output_accum(
    stream_t(Convolution110_Output_acc_t) &in,
    stream_t(Convolution110_Output_acc_t) &out
) {

#pragma HLS INLINE OFF

    accum<
        CONVOLUTION110_OUTPUT_ACCUM_BATCH_SIZE,
        CONVOLUTION110_OUTPUT_ACCUM_ROWS,
        CONVOLUTION110_OUTPUT_ACCUM_COLS,
        CONVOLUTION110_OUTPUT_ACCUM_CHANNELS,
        CONVOLUTION110_OUTPUT_ACCUM_FILTERS,
        CONVOLUTION110_OUTPUT_ACCUM_GROUPS,
        Convolution110_Output_acc_t
    >(in,out);

}

void Convolution110_Output_glue(
    stream_t(Convolution110_Output_acc_t) in[CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP][CONVOLUTION110_OUTPUT_COARSE_OUT],
    stream_t(Convolution110_Output_output_t) out[CONVOLUTION110_OUTPUT_COARSE_OUT]
) {

#pragma HLS INLINE OFF

    glue<
        CONVOLUTION110_OUTPUT_GLUE_BATCH_SIZE,
        CONVOLUTION110_OUTPUT_GLUE_ROWS,
        CONVOLUTION110_OUTPUT_GLUE_COLS,
        CONVOLUTION110_OUTPUT_GLUE_FILTERS,
        CONVOLUTION110_OUTPUT_GLUE_COARSE_IN,
        CONVOLUTION110_OUTPUT_GLUE_COARSE_OUT,
        CONVOLUTION110_OUTPUT_GLUE_COARSE_GROUP,
        Convolution110_Output_acc_t,
        Convolution110_Output_output_t
    >(in,out);

}

void Convolution110_Output_bias(
    const Convolution110_Output_biases_t biases[CONVOLUTION110_OUTPUT_BIAS_FILTERS],
    stream_t(Convolution110_Output_output_t) &in,
    stream_t(Convolution110_Output_output_t) &out
) {

#pragma HLS INLINE OFF

    bias<
        CONVOLUTION110_OUTPUT_BIAS_BATCH_SIZE,
        CONVOLUTION110_OUTPUT_BIAS_ROWS,
        CONVOLUTION110_OUTPUT_BIAS_COLS,
        CONVOLUTION110_OUTPUT_BIAS_FILTERS,
        Convolution110_Output_output_t,
        Convolution110_Output_biases_t
    >(in,biases,out);

}

void Convolution110_Output(
    const Convolution110_Output_weight_t weights[CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP][CONVOLUTION110_OUTPUT_COARSE_OUT][DIVIDE(CONVOLUTION110_OUTPUT_WEIGHTS,CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP*CONVOLUTION110_OUTPUT_COARSE_OUT*CONVOLUTION110_OUTPUT_KERNEL_SIZE_X*CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y)][CONVOLUTION110_OUTPUT_KERNEL_SIZE_X][CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y],
#if CONVOLUTION110_OUTPUT_HAS_BIAS == 1
    const Convolution110_Output_biases_t biases[CONVOLUTION110_OUTPUT_COARSE_OUT][CONVOLUTION110_OUTPUT_BIAS_FILTERS],
#endif
    stream_t(Convolution110_Output_input_t)  in[CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP],
    stream_t(Convolution110_Output_output_t) out[CONVOLUTION110_OUTPUT_COARSE_OUT*CONVOLUTION110_OUTPUT_COARSE_GROUP],
    int mode
)
{

#pragma HLS INLINE OFF

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

#pragma HLS DATAFLOW
#pragma HLS stable variable=weights

#if CONVOLUTION110_OUTPUT_KERNEL_SIZE_X >= 1 || CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y >= 1
    stream_t(Convolution110_Output_input_t) sw_out[CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP][CONVOLUTION110_OUTPUT_KERNEL_SIZE_X][CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y];
    #pragma HLS STREAM variable=sw_out
    #pragma HLS ARRAY_PARTITION variable=sw_out complete dim=0
#endif

#if CONVOLUTION110_OUTPUT_KERNEL_SIZE_X == 1 && CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y == 1
    stream_t(Convolution110_Output_input_t) fork_out[CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP][CONVOLUTION110_OUTPUT_COARSE_OUT];
#else
    stream_t(Convolution110_Output_input_t) fork_out[CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP][CONVOLUTION110_OUTPUT_COARSE_OUT][CONVOLUTION110_OUTPUT_KERNEL_SIZE_X][CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y];
#endif
    #pragma HLS STREAM variable=fork_out
    #pragma HLS ARRAY_PARTITION variable=fork_out complete dim=0

    stream_t(Convolution110_Output_acc_t) conv_out[CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP][CONVOLUTION110_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=conv_out
    #pragma HLS ARRAY_PARTITION variable=conv_out complete dim=0

#if CONVOLUTION110_OUTPUT_ACCUM_CHANNELS > 1
    stream_t(Convolution110_Output_acc_t) accum_out[CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP][CONVOLUTION110_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=accum_out
    #pragma HLS ARRAY_PARTITION variable=accum_out complete dim=0
#endif

#if CONVOLUTION110_OUTPUT_HAS_BIAS == 1
    stream_t(Convolution110_Output_output_t) glue_out[CONVOLUTION110_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=glue_out
    #pragma HLS ARRAY_PARTITION variable=glue_out complete dim=0
#endif

    Convolution110_Output_coarse_in_loop: for(unsigned int i=0;i<CONVOLUTION110_OUTPUT_COARSE_IN*CONVOLUTION110_OUTPUT_COARSE_GROUP;i++) {
        #pragma HLS unroll
#if CONVOLUTION110_OUTPUT_KERNEL_SIZE_X == 1 && CONVOLUTION110_OUTPUT_KERNEL_SIZE_Y == 1
        Convolution110_Output_fork(in[i], fork_out[i]);
#else
        Convolution110_Output_sliding_window(in[i], sw_out[i]);
        Convolution110_Output_fork(sw_out[i], fork_out[i]);
#endif
        Convolution110_Output_coarse_out_loop: for(unsigned int j=0;j<CONVOLUTION110_OUTPUT_COARSE_OUT;j++) {
            #pragma HLS unroll
            Convolution110_Output_conv(weights[i][j], fork_out[i][j], conv_out[i][j]);
#if CONVOLUTION110_OUTPUT_ACCUM_CHANNELS > 1
            Convolution110_Output_accum(conv_out[i][j], accum_out[i][j]);
#endif
        }
    }

#if CONVOLUTION110_OUTPUT_ACCUM_CHANNELS > 1
#if CONVOLUTION110_OUTPUT_HAS_BIAS == 1

    Convolution110_Output_glue(accum_out, glue_out);

    Convolution110_Output_coarse_out_bias_loop: for(unsigned int i=0;i<CONVOLUTION110_OUTPUT_COARSE_OUT;i++) {
        #pragma HLS unroll
        Convolution110_Output_bias(biases[i], glue_out[i], out[i]);
    }

#else

    Convolution110_Output_glue(accum_out, out);

#endif
#else
#if CONVOLUTION110_OUTPUT_HAS_BIAS == 1

    Convolution110_Output_glue(conv_out, glue_out);

    Convolution110_Output_coarse_out_bias_loop: for(unsigned int i=0;i<CONVOLUTION110_OUTPUT_COARSE_OUT;i++) {
        #pragma HLS unroll
        Convolution110_Output_bias(biases[i], glue_out[i], out[i]);
    }

#else

    Convolution110_Output_glue(conv_out, out);

#endif
#endif

}

