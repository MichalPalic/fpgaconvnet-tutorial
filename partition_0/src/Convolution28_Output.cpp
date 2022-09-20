#include "Convolution28_Output.hpp"

void Convolution28_Output_sliding_window(
    stream_t(Convolution28_Output_input_t)  &in,
    stream_t(Convolution28_Output_output_t) out[CONVOLUTION28_OUTPUT_KERNEL_SIZE_X][CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y]
) {

#pragma HLS INLINE OFF

    sliding_window<
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_BATCH_SIZE,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_ROWS,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_COLS,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_CHANNELS,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_PAD_TOP,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_PAD_RIGHT,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_PAD_BOTTOM,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_PAD_LEFT,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_STRIDE_X,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_STRIDE_Y,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_KERNEL_SIZE_X,
        CONVOLUTION28_OUTPUT_SLIDING_WINDOW_KERNEL_SIZE_Y,
        Convolution28_Output_input_t
    >(in,out);

}

void Convolution28_Output_fork(
#if CONVOLUTION28_OUTPUT_KERNEL_SIZE_X == 1 && CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y == 1
    stream_t(Convolution28_Output_input_t)  &in,
    stream_t(Convolution28_Output_output_t) out[CONVOLUTION28_OUTPUT_COARSE_OUT]
#else
    stream_t(Convolution28_Output_input_t)  in[CONVOLUTION28_OUTPUT_KERNEL_SIZE_X][CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y],
    stream_t(Convolution28_Output_output_t) out[CONVOLUTION28_OUTPUT_COARSE_OUT][CONVOLUTION28_OUTPUT_KERNEL_SIZE_X][CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y]
#endif
) {

#pragma HLS INLINE OFF

    fork<
        CONVOLUTION28_OUTPUT_FORK_BATCH_SIZE,
        CONVOLUTION28_OUTPUT_FORK_ROWS,
        CONVOLUTION28_OUTPUT_FORK_COLS,
        CONVOLUTION28_OUTPUT_FORK_CHANNELS,
        CONVOLUTION28_OUTPUT_FORK_COARSE,
#if CONVOLUTION28_OUTPUT_FORK_KERNEL_SIZE_X > 1 || CONVOLUTION28_OUTPUT_FORK_KERNEL_SIZE_Y > 1
        CONVOLUTION28_OUTPUT_FORK_KERNEL_SIZE_X,
        CONVOLUTION28_OUTPUT_FORK_KERNEL_SIZE_Y,
#endif
        Convolution28_Output_input_t
    >(in,out);

}

void Convolution28_Output_conv(
    const Convolution28_Output_weight_t weights[DIVIDE(CONVOLUTION28_OUTPUT_WEIGHTS,CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP*CONVOLUTION28_OUTPUT_COARSE_OUT*CONVOLUTION28_OUTPUT_KERNEL_SIZE_X*CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y)][CONVOLUTION28_OUTPUT_KERNEL_SIZE_X][CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y],
#if CONVOLUTION28_OUTPUT_KERNEL_SIZE_X == 1 && CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y == 1
    stream_t(Convolution28_Output_input_t) &in,
#else
    stream_t(Convolution28_Output_input_t)  in[CONVOLUTION28_OUTPUT_KERNEL_SIZE_X][CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y],
#endif
    stream_t(Convolution28_Output_acc_t) &out
) {

#pragma HLS INLINE OFF

    conv<
        CONVOLUTION28_OUTPUT_CONV_BATCH_SIZE,
        CONVOLUTION28_OUTPUT_CONV_ROWS,
        CONVOLUTION28_OUTPUT_CONV_COLS,
        CONVOLUTION28_OUTPUT_CONV_CHANNELS,
        CONVOLUTION28_OUTPUT_CONV_FILTERS,
        CONVOLUTION28_OUTPUT_CONV_GROUPS,
#if (CONVOLUTION28_OUTPUT_CONV_KERNEL_SIZE_X > 1) || (CONVOLUTION28_OUTPUT_CONV_KERNEL_SIZE_Y > 1)
        CONVOLUTION28_OUTPUT_CONV_FINE,
        CONVOLUTION28_OUTPUT_CONV_KERNEL_SIZE_X,
        CONVOLUTION28_OUTPUT_CONV_KERNEL_SIZE_Y,
#endif
        Convolution28_Output_input_t,
        Convolution28_Output_weight_t,
        Convolution28_Output_acc_t
    >(in,weights,out);

}

void Convolution28_Output_accum(
    stream_t(Convolution28_Output_acc_t) &in,
    stream_t(Convolution28_Output_acc_t) &out
) {

#pragma HLS INLINE OFF

    accum<
        CONVOLUTION28_OUTPUT_ACCUM_BATCH_SIZE,
        CONVOLUTION28_OUTPUT_ACCUM_ROWS,
        CONVOLUTION28_OUTPUT_ACCUM_COLS,
        CONVOLUTION28_OUTPUT_ACCUM_CHANNELS,
        CONVOLUTION28_OUTPUT_ACCUM_FILTERS,
        CONVOLUTION28_OUTPUT_ACCUM_GROUPS,
        Convolution28_Output_acc_t
    >(in,out);

}

void Convolution28_Output_glue(
    stream_t(Convolution28_Output_acc_t) in[CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP][CONVOLUTION28_OUTPUT_COARSE_OUT],
    stream_t(Convolution28_Output_output_t) out[CONVOLUTION28_OUTPUT_COARSE_OUT]
) {

#pragma HLS INLINE OFF

    glue<
        CONVOLUTION28_OUTPUT_GLUE_BATCH_SIZE,
        CONVOLUTION28_OUTPUT_GLUE_ROWS,
        CONVOLUTION28_OUTPUT_GLUE_COLS,
        CONVOLUTION28_OUTPUT_GLUE_FILTERS,
        CONVOLUTION28_OUTPUT_GLUE_COARSE_IN,
        CONVOLUTION28_OUTPUT_GLUE_COARSE_OUT,
        CONVOLUTION28_OUTPUT_GLUE_COARSE_GROUP,
        Convolution28_Output_acc_t,
        Convolution28_Output_output_t
    >(in,out);

}

void Convolution28_Output_bias(
    const Convolution28_Output_biases_t biases[CONVOLUTION28_OUTPUT_BIAS_FILTERS],
    stream_t(Convolution28_Output_output_t) &in,
    stream_t(Convolution28_Output_output_t) &out
) {

#pragma HLS INLINE OFF

    bias<
        CONVOLUTION28_OUTPUT_BIAS_BATCH_SIZE,
        CONVOLUTION28_OUTPUT_BIAS_ROWS,
        CONVOLUTION28_OUTPUT_BIAS_COLS,
        CONVOLUTION28_OUTPUT_BIAS_FILTERS,
        Convolution28_Output_output_t,
        Convolution28_Output_biases_t
    >(in,biases,out);

}

void Convolution28_Output(
    const Convolution28_Output_weight_t weights[CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP][CONVOLUTION28_OUTPUT_COARSE_OUT][DIVIDE(CONVOLUTION28_OUTPUT_WEIGHTS,CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP*CONVOLUTION28_OUTPUT_COARSE_OUT*CONVOLUTION28_OUTPUT_KERNEL_SIZE_X*CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y)][CONVOLUTION28_OUTPUT_KERNEL_SIZE_X][CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y],
#if CONVOLUTION28_OUTPUT_HAS_BIAS == 1
    const Convolution28_Output_biases_t biases[CONVOLUTION28_OUTPUT_COARSE_OUT][CONVOLUTION28_OUTPUT_BIAS_FILTERS],
#endif
    stream_t(Convolution28_Output_input_t)  in[CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP],
    stream_t(Convolution28_Output_output_t) out[CONVOLUTION28_OUTPUT_COARSE_OUT*CONVOLUTION28_OUTPUT_COARSE_GROUP],
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

#if CONVOLUTION28_OUTPUT_KERNEL_SIZE_X >= 1 || CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y >= 1
    stream_t(Convolution28_Output_input_t) sw_out[CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP][CONVOLUTION28_OUTPUT_KERNEL_SIZE_X][CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y];
    #pragma HLS STREAM variable=sw_out
    #pragma HLS ARRAY_PARTITION variable=sw_out complete dim=0
#endif

#if CONVOLUTION28_OUTPUT_KERNEL_SIZE_X == 1 && CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y == 1
    stream_t(Convolution28_Output_input_t) fork_out[CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP][CONVOLUTION28_OUTPUT_COARSE_OUT];
#else
    stream_t(Convolution28_Output_input_t) fork_out[CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP][CONVOLUTION28_OUTPUT_COARSE_OUT][CONVOLUTION28_OUTPUT_KERNEL_SIZE_X][CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y];
#endif
    #pragma HLS STREAM variable=fork_out
    #pragma HLS ARRAY_PARTITION variable=fork_out complete dim=0

    stream_t(Convolution28_Output_acc_t) conv_out[CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP][CONVOLUTION28_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=conv_out
    #pragma HLS ARRAY_PARTITION variable=conv_out complete dim=0

#if CONVOLUTION28_OUTPUT_ACCUM_CHANNELS > 1
    stream_t(Convolution28_Output_acc_t) accum_out[CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP][CONVOLUTION28_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=accum_out
    #pragma HLS ARRAY_PARTITION variable=accum_out complete dim=0
#endif

#if CONVOLUTION28_OUTPUT_HAS_BIAS == 1
    stream_t(Convolution28_Output_output_t) glue_out[CONVOLUTION28_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=glue_out
    #pragma HLS ARRAY_PARTITION variable=glue_out complete dim=0
#endif

    Convolution28_Output_coarse_in_loop: for(unsigned int i=0;i<CONVOLUTION28_OUTPUT_COARSE_IN*CONVOLUTION28_OUTPUT_COARSE_GROUP;i++) {
        #pragma HLS unroll
#if CONVOLUTION28_OUTPUT_KERNEL_SIZE_X == 1 && CONVOLUTION28_OUTPUT_KERNEL_SIZE_Y == 1
        Convolution28_Output_fork(in[i], fork_out[i]);
#else
        Convolution28_Output_sliding_window(in[i], sw_out[i]);
        Convolution28_Output_fork(sw_out[i], fork_out[i]);
#endif
        Convolution28_Output_coarse_out_loop: for(unsigned int j=0;j<CONVOLUTION28_OUTPUT_COARSE_OUT;j++) {
            #pragma HLS unroll
            Convolution28_Output_conv(weights[i][j], fork_out[i][j], conv_out[i][j]);
#if CONVOLUTION28_OUTPUT_ACCUM_CHANNELS > 1
            Convolution28_Output_accum(conv_out[i][j], accum_out[i][j]);
#endif
        }
    }

#if CONVOLUTION28_OUTPUT_ACCUM_CHANNELS > 1
#if CONVOLUTION28_OUTPUT_HAS_BIAS == 1

    Convolution28_Output_glue(accum_out, glue_out);

    Convolution28_Output_coarse_out_bias_loop: for(unsigned int i=0;i<CONVOLUTION28_OUTPUT_COARSE_OUT;i++) {
        #pragma HLS unroll
        Convolution28_Output_bias(biases[i], glue_out[i], out[i]);
    }

#else

    Convolution28_Output_glue(accum_out, out);

#endif
#else
#if CONVOLUTION28_OUTPUT_HAS_BIAS == 1

    Convolution28_Output_glue(conv_out, glue_out);

    Convolution28_Output_coarse_out_bias_loop: for(unsigned int i=0;i<CONVOLUTION28_OUTPUT_COARSE_OUT;i++) {
        #pragma HLS unroll
        Convolution28_Output_bias(biases[i], glue_out[i], out[i]);
    }

#else

    Convolution28_Output_glue(conv_out, out);

#endif
#endif

}

