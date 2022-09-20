#include "import_conv4last_BiasAdd.hpp"

void import_conv4last_BiasAdd_sliding_window(
    stream_t(import_conv4last_BiasAdd_input_t)  &in,
    stream_t(import_conv4last_BiasAdd_output_t) out[IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y]
) {

#pragma HLS INLINE OFF

    sliding_window<
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_BATCH_SIZE,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_ROWS,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_COLS,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_CHANNELS,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_PAD_TOP,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_PAD_RIGHT,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_PAD_BOTTOM,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_PAD_LEFT,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_STRIDE_X,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_STRIDE_Y,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_KERNEL_SIZE_X,
        IMPORT_CONV4LAST_BIASADD_SLIDING_WINDOW_KERNEL_SIZE_Y,
        import_conv4last_BiasAdd_input_t
    >(in,out);

}

void import_conv4last_BiasAdd_fork(
#if IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X == 1 && IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y == 1
    stream_t(import_conv4last_BiasAdd_input_t)  &in,
    stream_t(import_conv4last_BiasAdd_output_t) out[IMPORT_CONV4LAST_BIASADD_COARSE_OUT]
#else
    stream_t(import_conv4last_BiasAdd_input_t)  in[IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y],
    stream_t(import_conv4last_BiasAdd_output_t) out[IMPORT_CONV4LAST_BIASADD_COARSE_OUT][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y]
#endif
) {

#pragma HLS INLINE OFF

    fork<
        IMPORT_CONV4LAST_BIASADD_FORK_BATCH_SIZE,
        IMPORT_CONV4LAST_BIASADD_FORK_ROWS,
        IMPORT_CONV4LAST_BIASADD_FORK_COLS,
        IMPORT_CONV4LAST_BIASADD_FORK_CHANNELS,
        IMPORT_CONV4LAST_BIASADD_FORK_COARSE,
#if IMPORT_CONV4LAST_BIASADD_FORK_KERNEL_SIZE_X > 1 || IMPORT_CONV4LAST_BIASADD_FORK_KERNEL_SIZE_Y > 1
        IMPORT_CONV4LAST_BIASADD_FORK_KERNEL_SIZE_X,
        IMPORT_CONV4LAST_BIASADD_FORK_KERNEL_SIZE_Y,
#endif
        import_conv4last_BiasAdd_input_t
    >(in,out);

}

void import_conv4last_BiasAdd_conv(
    const import_conv4last_BiasAdd_weight_t weights[DIVIDE(IMPORT_CONV4LAST_BIASADD_WEIGHTS,IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP*IMPORT_CONV4LAST_BIASADD_COARSE_OUT*IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X*IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y)][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y],
#if IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X == 1 && IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y == 1
    stream_t(import_conv4last_BiasAdd_input_t) &in,
#else
    stream_t(import_conv4last_BiasAdd_input_t)  in[IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y],
#endif
    stream_t(import_conv4last_BiasAdd_acc_t) &out
) {

#pragma HLS INLINE OFF

    conv<
        IMPORT_CONV4LAST_BIASADD_CONV_BATCH_SIZE,
        IMPORT_CONV4LAST_BIASADD_CONV_ROWS,
        IMPORT_CONV4LAST_BIASADD_CONV_COLS,
        IMPORT_CONV4LAST_BIASADD_CONV_CHANNELS,
        IMPORT_CONV4LAST_BIASADD_CONV_FILTERS,
        IMPORT_CONV4LAST_BIASADD_CONV_GROUPS,
#if (IMPORT_CONV4LAST_BIASADD_CONV_KERNEL_SIZE_X > 1) || (IMPORT_CONV4LAST_BIASADD_CONV_KERNEL_SIZE_Y > 1)
        IMPORT_CONV4LAST_BIASADD_CONV_FINE,
        IMPORT_CONV4LAST_BIASADD_CONV_KERNEL_SIZE_X,
        IMPORT_CONV4LAST_BIASADD_CONV_KERNEL_SIZE_Y,
#endif
        import_conv4last_BiasAdd_input_t,
        import_conv4last_BiasAdd_weight_t,
        import_conv4last_BiasAdd_acc_t
    >(in,weights,out);

}

void import_conv4last_BiasAdd_accum(
    stream_t(import_conv4last_BiasAdd_acc_t) &in,
    stream_t(import_conv4last_BiasAdd_acc_t) &out
) {

#pragma HLS INLINE OFF

    accum<
        IMPORT_CONV4LAST_BIASADD_ACCUM_BATCH_SIZE,
        IMPORT_CONV4LAST_BIASADD_ACCUM_ROWS,
        IMPORT_CONV4LAST_BIASADD_ACCUM_COLS,
        IMPORT_CONV4LAST_BIASADD_ACCUM_CHANNELS,
        IMPORT_CONV4LAST_BIASADD_ACCUM_FILTERS,
        IMPORT_CONV4LAST_BIASADD_ACCUM_GROUPS,
        import_conv4last_BiasAdd_acc_t
    >(in,out);

}

void import_conv4last_BiasAdd_glue(
    stream_t(import_conv4last_BiasAdd_acc_t) in[IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP][IMPORT_CONV4LAST_BIASADD_COARSE_OUT],
    stream_t(import_conv4last_BiasAdd_output_t) out[IMPORT_CONV4LAST_BIASADD_COARSE_OUT]
) {

#pragma HLS INLINE OFF

    glue<
        IMPORT_CONV4LAST_BIASADD_GLUE_BATCH_SIZE,
        IMPORT_CONV4LAST_BIASADD_GLUE_ROWS,
        IMPORT_CONV4LAST_BIASADD_GLUE_COLS,
        IMPORT_CONV4LAST_BIASADD_GLUE_FILTERS,
        IMPORT_CONV4LAST_BIASADD_GLUE_COARSE_IN,
        IMPORT_CONV4LAST_BIASADD_GLUE_COARSE_OUT,
        IMPORT_CONV4LAST_BIASADD_GLUE_COARSE_GROUP,
        import_conv4last_BiasAdd_acc_t,
        import_conv4last_BiasAdd_output_t
    >(in,out);

}

void import_conv4last_BiasAdd_bias(
    const import_conv4last_BiasAdd_biases_t biases[IMPORT_CONV4LAST_BIASADD_BIAS_FILTERS],
    stream_t(import_conv4last_BiasAdd_output_t) &in,
    stream_t(import_conv4last_BiasAdd_output_t) &out
) {

#pragma HLS INLINE OFF

    bias<
        IMPORT_CONV4LAST_BIASADD_BIAS_BATCH_SIZE,
        IMPORT_CONV4LAST_BIASADD_BIAS_ROWS,
        IMPORT_CONV4LAST_BIASADD_BIAS_COLS,
        IMPORT_CONV4LAST_BIASADD_BIAS_FILTERS,
        import_conv4last_BiasAdd_output_t,
        import_conv4last_BiasAdd_biases_t
    >(in,biases,out);

}

void import_conv4last_BiasAdd(
    const import_conv4last_BiasAdd_weight_t weights[IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP][IMPORT_CONV4LAST_BIASADD_COARSE_OUT][DIVIDE(IMPORT_CONV4LAST_BIASADD_WEIGHTS,IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP*IMPORT_CONV4LAST_BIASADD_COARSE_OUT*IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X*IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y)][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y],
#if IMPORT_CONV4LAST_BIASADD_HAS_BIAS == 1
    const import_conv4last_BiasAdd_biases_t biases[IMPORT_CONV4LAST_BIASADD_COARSE_OUT][IMPORT_CONV4LAST_BIASADD_BIAS_FILTERS],
#endif
    stream_t(import_conv4last_BiasAdd_input_t)  in[IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP],
    stream_t(import_conv4last_BiasAdd_output_t) out[IMPORT_CONV4LAST_BIASADD_COARSE_OUT*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP],
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

#if IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X >= 1 || IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y >= 1
    stream_t(import_conv4last_BiasAdd_input_t) sw_out[IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y];
    #pragma HLS STREAM variable=sw_out
    #pragma HLS ARRAY_PARTITION variable=sw_out complete dim=0
#endif

#if IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X == 1 && IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y == 1
    stream_t(import_conv4last_BiasAdd_input_t) fork_out[IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP][IMPORT_CONV4LAST_BIASADD_COARSE_OUT];
#else
    stream_t(import_conv4last_BiasAdd_input_t) fork_out[IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP][IMPORT_CONV4LAST_BIASADD_COARSE_OUT][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X][IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y];
#endif
    #pragma HLS STREAM variable=fork_out
    #pragma HLS ARRAY_PARTITION variable=fork_out complete dim=0

    stream_t(import_conv4last_BiasAdd_acc_t) conv_out[IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP][IMPORT_CONV4LAST_BIASADD_COARSE_OUT];
    #pragma HLS STREAM variable=conv_out
    #pragma HLS ARRAY_PARTITION variable=conv_out complete dim=0

#if IMPORT_CONV4LAST_BIASADD_ACCUM_CHANNELS > 1
    stream_t(import_conv4last_BiasAdd_acc_t) accum_out[IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP][IMPORT_CONV4LAST_BIASADD_COARSE_OUT];
    #pragma HLS STREAM variable=accum_out
    #pragma HLS ARRAY_PARTITION variable=accum_out complete dim=0
#endif

#if IMPORT_CONV4LAST_BIASADD_HAS_BIAS == 1
    stream_t(import_conv4last_BiasAdd_output_t) glue_out[IMPORT_CONV4LAST_BIASADD_COARSE_OUT];
    #pragma HLS STREAM variable=glue_out
    #pragma HLS ARRAY_PARTITION variable=glue_out complete dim=0
#endif

    import_conv4last_BiasAdd_coarse_in_loop: for(unsigned int i=0;i<IMPORT_CONV4LAST_BIASADD_COARSE_IN*IMPORT_CONV4LAST_BIASADD_COARSE_GROUP;i++) {
        #pragma HLS unroll
#if IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_X == 1 && IMPORT_CONV4LAST_BIASADD_KERNEL_SIZE_Y == 1
        import_conv4last_BiasAdd_fork(in[i], fork_out[i]);
#else
        import_conv4last_BiasAdd_sliding_window(in[i], sw_out[i]);
        import_conv4last_BiasAdd_fork(sw_out[i], fork_out[i]);
#endif
        import_conv4last_BiasAdd_coarse_out_loop: for(unsigned int j=0;j<IMPORT_CONV4LAST_BIASADD_COARSE_OUT;j++) {
            #pragma HLS unroll
            import_conv4last_BiasAdd_conv(weights[i][j], fork_out[i][j], conv_out[i][j]);
#if IMPORT_CONV4LAST_BIASADD_ACCUM_CHANNELS > 1
            import_conv4last_BiasAdd_accum(conv_out[i][j], accum_out[i][j]);
#endif
        }
    }

#if IMPORT_CONV4LAST_BIASADD_ACCUM_CHANNELS > 1
#if IMPORT_CONV4LAST_BIASADD_HAS_BIAS == 1

    import_conv4last_BiasAdd_glue(accum_out, glue_out);

    import_conv4last_BiasAdd_coarse_out_bias_loop: for(unsigned int i=0;i<IMPORT_CONV4LAST_BIASADD_COARSE_OUT;i++) {
        #pragma HLS unroll
        import_conv4last_BiasAdd_bias(biases[i], glue_out[i], out[i]);
    }

#else

    import_conv4last_BiasAdd_glue(accum_out, out);

#endif
#else
#if IMPORT_CONV4LAST_BIASADD_HAS_BIAS == 1

    import_conv4last_BiasAdd_glue(conv_out, glue_out);

    import_conv4last_BiasAdd_coarse_out_bias_loop: for(unsigned int i=0;i<IMPORT_CONV4LAST_BIASADD_COARSE_OUT;i++) {
        #pragma HLS unroll
        import_conv4last_BiasAdd_bias(biases[i], glue_out[i], out[i]);
    }

#else

    import_conv4last_BiasAdd_glue(conv_out, out);

#endif
#endif

}

