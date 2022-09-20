#include "Plus214_Output.hpp"

void Plus214_Output_fork(
    stream_t(Plus214_Output_input_t)  &in,
    stream_t(Plus214_Output_output_t) out[PLUS214_OUTPUT_COARSE_OUT]
) {

#pragma HLS INLINE OFF

    fork<
        PLUS214_OUTPUT_FORK_BATCH_SIZE,
        PLUS214_OUTPUT_FORK_ROWS,
        PLUS214_OUTPUT_FORK_COLS,
        PLUS214_OUTPUT_FORK_CHANNELS,
        PLUS214_OUTPUT_FORK_COARSE,
#if PLUS214_OUTPUT_FORK_KERNEL_SIZE_X > 1 || PLUS214_OUTPUT_FORK_KERNEL_SIZE_Y > 1
        PLUS214_OUTPUT_FORK_KERNEL_SIZE_X,
        PLUS214_OUTPUT_FORK_KERNEL_SIZE_Y,
#endif
        Plus214_Output_input_t
    >(in,out);

}

void Plus214_Output_conv(
    const Plus214_Output_weight_t weights[DIVIDE(PLUS214_OUTPUT_WEIGHTS,PLUS214_OUTPUT_COARSE_IN*PLUS214_OUTPUT_COARSE_OUT)][1][1],
    stream_t(Plus214_Output_input_t) &in,
    stream_t(Plus214_Output_acc_t) &out
) {

#pragma HLS INLINE OFF

    conv<
        PLUS214_OUTPUT_CONV_BATCH_SIZE,
        PLUS214_OUTPUT_CONV_ROWS,
        PLUS214_OUTPUT_CONV_COLS,
        PLUS214_OUTPUT_CONV_CHANNELS,
        PLUS214_OUTPUT_CONV_FILTERS,
        PLUS214_OUTPUT_CONV_GROUPS,
#if (PLUS214_OUTPUT_CONV_KERNEL_SIZE_X > 1) || (PLUS214_OUTPUT_CONV_KERNEL_SIZE_Y > 1)
        PLUS214_OUTPUT_CONV_FINE,
        PLUS214_OUTPUT_CONV_KERNEL_SIZE_X,
        PLUS214_OUTPUT_CONV_KERNEL_SIZE_Y,
#endif
        Plus214_Output_input_t,
        Plus214_Output_weight_t,
        Plus214_Output_acc_t
    >(in,weights,out);

}

void Plus214_Output_accum(
    stream_t(Plus214_Output_acc_t) &in,
    stream_t(Plus214_Output_acc_t) &out
) {

#pragma HLS INLINE OFF

    accum<
        PLUS214_OUTPUT_ACCUM_BATCH_SIZE,
        PLUS214_OUTPUT_ACCUM_ROWS,
        PLUS214_OUTPUT_ACCUM_COLS,
        PLUS214_OUTPUT_ACCUM_CHANNELS,
        PLUS214_OUTPUT_ACCUM_FILTERS,
        PLUS214_OUTPUT_ACCUM_GROUPS,
        Plus214_Output_acc_t
    >(in,out);

}

void Plus214_Output_glue(
    stream_t(Plus214_Output_acc_t) in[PLUS214_OUTPUT_COARSE_IN][PLUS214_OUTPUT_COARSE_OUT],
    stream_t(Plus214_Output_output_t) out[PLUS214_OUTPUT_COARSE_OUT]
) {

#pragma HLS INLINE OFF

    glue<
        PLUS214_OUTPUT_GLUE_BATCH_SIZE,
        PLUS214_OUTPUT_GLUE_ROWS,
        PLUS214_OUTPUT_GLUE_COLS,
        PLUS214_OUTPUT_GLUE_FILTERS,
        PLUS214_OUTPUT_GLUE_COARSE_IN,
        PLUS214_OUTPUT_GLUE_COARSE_OUT,
        PLUS214_OUTPUT_GLUE_COARSE_GROUP,
        Plus214_Output_acc_t,
        Plus214_Output_output_t
    >(in,out);

}

void Plus214_Output_bias(
    const Plus214_Output_biases_t biases[PLUS214_OUTPUT_BIAS_FILTERS],
    stream_t(Plus214_Output_output_t) &in,
    stream_t(Plus214_Output_output_t) &out
) {

#pragma HLS INLINE OFF

    bias<
        PLUS214_OUTPUT_BIAS_BATCH_SIZE,
        PLUS214_OUTPUT_BIAS_ROWS,
        PLUS214_OUTPUT_BIAS_COLS,
        PLUS214_OUTPUT_BIAS_FILTERS,
        Plus214_Output_output_t,
        Plus214_Output_biases_t
    >(in,biases,out);

}

void Plus214_Output(
    const Plus214_Output_weight_t weights[PLUS214_OUTPUT_COARSE_IN][PLUS214_OUTPUT_COARSE_OUT][DIVIDE(PLUS214_OUTPUT_WEIGHTS,PLUS214_OUTPUT_COARSE_IN*PLUS214_OUTPUT_COARSE_OUT)][1][1],
#if PLUS214_OUTPUT_HAS_BIAS == 1
    const Plus214_Output_biases_t biases[PLUS214_OUTPUT_COARSE_OUT][PLUS214_OUTPUT_BIAS_FILTERS],
#endif
    stream_t(Plus214_Output_input_t) in[PLUS214_OUTPUT_COARSE_IN],
    stream_t(Plus214_Output_output_t) out[PLUS214_OUTPUT_COARSE_OUT],
    int mode
)
{

#pragma HLS INLINE OFF

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

#pragma HLS DATAFLOW

    stream_t(Plus214_Output_input_t) fork_out[PLUS214_OUTPUT_COARSE_IN][PLUS214_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=fork_out
    #pragma HLS ARRAY_PARTITION variable=fork_out complete dim=0

    stream_t(Plus214_Output_acc_t) conv_out[PLUS214_OUTPUT_COARSE_IN][PLUS214_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=conv_out
    #pragma HLS ARRAY_PARTITION variable=conv_out complete dim=0

#if PLUS214_OUTPUT_ACCUM_CHANNELS > 1
    stream_t(Plus214_Output_acc_t) accum_out[PLUS214_OUTPUT_COARSE_IN][PLUS214_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=accum_out
    #pragma HLS ARRAY_PARTITION variable=accum_out complete dim=0
#endif

#if PLUS214_OUTPUT_HAS_BIAS == 1
    stream_t(Plus214_Output_output_t) glue_out[PLUS214_OUTPUT_COARSE_OUT];
    #pragma HLS STREAM variable=glue_out
    #pragma HLS ARRAY_PARTITION variable=glue_out complete dim=0
#endif

    Plus214_Output_coarse_in_loop: for(int i=0;i<PLUS214_OUTPUT_COARSE_IN;i++) {
        #pragma HLS UNROLL

        Plus214_Output_fork(in[i], fork_out[i]);

        Plus214_Output_coarse_out_loop: for(int j=0;j<PLUS214_OUTPUT_COARSE_OUT;j++) {
            #pragma HLS UNROLL
            Plus214_Output_conv(weights[i][j], fork_out[i][j], conv_out[i][j]);
#if PLUS214_OUTPUT_ACCUM_CHANNELS > 1
            Plus214_Output_accum(conv_out[i][j], accum_out[i][j]);
#endif
        }
    }

#if PLUS214_OUTPUT_ACCUM_CHANNELS > 1
#if PLUS214_OUTPUT_HAS_BIAS == 1
    Plus214_Output_glue(accum_out, glue_out);
    Plus214_Output_coarse_out_bias_loop: for(unsigned int i=0;i<PLUS214_OUTPUT_COARSE_OUT;i++) {
        #pragma HLS unroll
        Plus214_Output_bias(biases[i], glue_out[i], out[i]);
    }
#else
    Plus214_Output_glue(accum_out, out);
#endif
#else
#if PLUS214_OUTPUT_HAS_BIAS == 1
    Plus214_Output_glue(conv_out, glue_out);
    Plus214_Output_coarse_out_bias_loop: for(unsigned int i=0;i<PLUS214_OUTPUT_COARSE_OUT;i++) {
        #pragma HLS unroll
        Plus214_Output_bias(biases[i], glue_out[i], out[i]);
    }
#else
    Plus214_Output_glue(conv_out, out);
#endif
#endif

}

