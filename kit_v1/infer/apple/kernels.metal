#include <metal_stdlib>
using namespace metal;

// Match the FP32 implementation's separate multiply and add operations.
#pragma clang fp contract(off)

// One SIMD group normalizes and rotates a Q/K head. Each lane owns four of
// the 128 channels, including its partner in the rotate-half operation.
kernel void kit_qknorm_rope128(
    device const float* qkv, device const float* q_weight, device const float* k_weight,
    device const float* cos_table, device const float* sin_table,
    device float* q_out, device float* k_out, constant long& sequence, constant long& heads,
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint s = uint(sequence), h = uint(heads), row = tid / 32;
    const uint head = row % h, token = (row / h) % s, batch = row / (h * s);
    const uint q_base = ((batch * s + token) * 3 * h + head) * 128;
    const uint k_base = q_base + h * 128;
    float q[4], k[4], q_squares = 0.f, k_squares = 0.f;
    for (uint j = 0; j < 4; ++j) {
        const uint i = lane + j * 32;
        q[j] = qkv[q_base + i];
        k[j] = qkv[k_base + i];
        q_squares += q[j] * q[j];
        k_squares += k[j] * k[j];
    }
    const float q_inv = rsqrt(simd_sum(q_squares) / 128.f + 1e-6f);
    const float k_inv = rsqrt(simd_sum(k_squares) / 128.f + 1e-6f);
    for (uint j = 0; j < 4; ++j) {
        const uint i = lane + j * 32;
        q[j] = (q[j] * q_inv) * q_weight[i];
        k[j] = (k[j] * k_inv) * k_weight[i];
    }
    const uint out_base = ((batch * h + head) * s + token) * 128;
    for (uint j = 0; j < 4; ++j) {
        const uint i = lane + j * 32, pos = token * 128 + i;
        const float sign = j < 2 ? -1.f : 1.f;
        q_out[out_base + i] = q[j] * cos_table[pos] + sign * q[(j + 2) % 4] * sin_table[pos];
        k_out[out_base + i] = k[j] * cos_table[pos] + sign * k[(j + 2) % 4] * sin_table[pos];
    }
}

// The final layer projects Q only at target positions. This kernel reads one
// projection (Q) or the first half of a packed projection (KV).
kernel void norm_rope_one(
    device const float* input, device const float* weight,
    device const float* cos_table, device const float* sin_table, device float* output,
    constant long& sequence, constant long& heads, constant long& projections,
    constant long& position_offset, uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint s = uint(sequence), h = uint(heads), row = tid / 32;
    const uint head = row % h, token = (row / h) % s, batch = row / (h * s);
    const uint base = ((batch * s + token) * uint(projections) * h + head) * 128;
    float values[4], squares = 0.f;
    for (uint j = 0; j < 4; ++j) {
        const float value = input[base + lane + j * 32];
        values[j] = value;
        squares += value * value;
    }
    const float inv = rsqrt(simd_sum(squares) / 128.f + 1e-6f);
    for (uint j = 0; j < 4; ++j) {
        values[j] = (values[j] * inv) * weight[lane + j * 32];
    }
    const uint out_base = ((batch * h + head) * s + token) * 128;
    for (uint j = 0; j < 4; ++j) {
        const uint i = lane + j * 32, pos = (token + uint(position_offset)) * 128 + i;
        const float sign = j < 2 ? -1.f : 1.f;
        output[out_base + i] = values[j] * cos_table[pos] + sign * values[(j + 2) % 4] * sin_table[pos];
    }
}

// [gate | up] come from one concatenated projection per token.
kernel void packed_f32(device const float* input, device float* output, constant long& width,
                       uint i [[thread_position_in_grid]]) {
    const uint w = uint(width), gate_index = (i / w) * (2 * w) + i % w;
    const float gate = input[gate_index];
    output[i] = (gate / (1.f + exp(-gate))) * input[gate_index + w];
}

// Update the residual, then apply the following sublayer's RMSNorm and AdaLN
// using the same row in registers. Context and target tokens have distinct modulation.
kernel void gated_norm_1024(
    device const float* x, device const float* delta,
    device const float* gate_ctx, device const float* gate_target,
    device const float* norm_ctx, device const float* norm_target,
    device float* saved, device float* normalized,
    constant long& sequence, constant long& context_length,
    constant long& gate_offset, constant long& norm_offset,
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid / 32, base = row * 1024;
    const bool context = row % uint(sequence) < uint(context_length);
    device const float* gate = (context ? gate_ctx : gate_target) + uint(gate_offset) * 1024;
    device const float* shift = (context ? norm_ctx : norm_target) + uint(norm_offset) * 1024;
    float values[32], squares = 0.f;
    for (uint j = 0; j < 32; ++j) {
        const uint i = lane + j * 32;
        const float value = x[base + i] + delta[base + i] * gate[i];
        values[j] = value;
        squares += value * value;
    }
    const float inv = rsqrt(simd_sum(squares) / 1024.f + 1e-6f);
    for (uint j = 0; j < 32; ++j) {
        const uint i = lane + j * 32;
        saved[base + i] = values[j];
        normalized[base + i] = (values[j] * inv) * (1.f + shift[1024 + i]) + shift[i];
    }
}
