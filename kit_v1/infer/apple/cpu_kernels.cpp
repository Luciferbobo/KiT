// KiT inference kernels. Each worker owns complete rows of its outputs.
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {
constexpr size_t kHeadDim = 128;
constexpr size_t kWorkers = 4;

void parallel_rows(size_t rows, void (^run)(size_t, size_t)) {
    const size_t workers = std::min(rows, kWorkers);
    dispatch_apply(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t worker) {
      run(rows * worker / workers, rows * (worker + 1) / workers);
    });
}

// Keep the FP32 reduction tree identical for every number of workers.
inline float sum_lanes(const float32x4_t (&sums)[4]) {
    return vaddvq_f32(vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3])));
}

void normalized_rope(const float *input, const float *weight, const float *cosines, const float *sines,
                     float epsilon, float *output) {
    float32x4_t sums[4] = {};
    for (size_t i = 0; i < kHeadDim; i += 16) {
        for (size_t lane = 0; lane < 4; ++lane) {
            const auto x = vld1q_f32(input + i + lane * 4);
            sums[lane] = vaddq_f32(sums[lane], vmulq_f32(x, x));
        }
    }
    const float scale = 1.f / std::sqrt(sum_lanes(sums) * (1.f / kHeadDim) + epsilon);
    for (size_t first = 0; first < kHeadDim / 2; first += 4) {
        const size_t second = first + kHeadDim / 2;
        const auto a =
            vmulq_f32(vmulq_n_f32(vld1q_f32(input + first), scale), vld1q_f32(weight + first));
        const auto b =
            vmulq_f32(vmulq_n_f32(vld1q_f32(input + second), scale), vld1q_f32(weight + second));
        vst1q_f32(output + first, vsubq_f32(vmulq_f32(a, vld1q_f32(cosines + first)),
                                            vmulq_f32(b, vld1q_f32(sines + first))));
        vst1q_f32(output + second, vaddq_f32(vmulq_f32(b, vld1q_f32(cosines + second)),
                                             vmulq_f32(a, vld1q_f32(sines + second))));
    }
}
} // namespace

// [batch, token, Q/K/V, head, 128] -> two [batch, token, head, 128] tensors.
extern "C" void kit_qk_rope(const float *qkv, const float *qw, const float *kw, const float *cosines,
                            const float *sines, float *q, float *k, size_t batch, size_t sequence,
                            size_t heads, float epsilon) {
    parallel_rows(batch * sequence * heads, ^(size_t begin, size_t end) {
      for (size_t row = begin; row < end; ++row) {
          const size_t token = row / heads;
          const size_t pos = (token % sequence) * kHeadDim;
          const float *raw_q = qkv + (token * 3 * heads + row % heads) * kHeadDim;
          normalized_rope(raw_q, qw, cosines + pos, sines + pos, epsilon, q + row * kHeadDim);
          normalized_rope(raw_q + heads * kHeadDim, kw, cosines + pos, sines + pos, epsilon,
                          k + row * kHeadDim);
      }
    });
}

// Fuse residual gating, RMSNorm and scale/shift. A sequence mixes context and target token parameters.
extern "C" void kit_gate_norm(const float *input, const float *delta, const float *context_gate,
                              const float *target_gate, const float *context_norm,
                              const float *target_norm, float *residual, float *hidden, size_t rows,
                              size_t sequence, size_t width, size_t prefix, size_t gate_offset,
                              size_t norm_offset) {
    parallel_rows(rows, ^(size_t begin, size_t end) {
      for (size_t row = begin; row < end; ++row) {
          const bool context = row % sequence < prefix;
          const float *gate = (context ? context_gate : target_gate) + gate_offset * width;
          const float *shift = (context ? context_norm : target_norm) + norm_offset * width;
          const float *scale = shift + width;
          const size_t base = row * width;
          float32x4_t sums[4] = {};
          for (size_t i = 0; i < width; i += 16) {
              for (size_t lane = 0; lane < 4; ++lane) {
                  const size_t j = i + lane * 4;
                  const auto update = vmulq_f32(vld1q_f32(delta + base + j), vld1q_f32(gate + j));
                  const auto z = vaddq_f32(vld1q_f32(input + base + j), update);
                  vst1q_f32(residual + base + j, z);
                  sums[lane] = vaddq_f32(sums[lane], vmulq_f32(z, z));
              }
          }
          const float inv_rms = 1.f / std::sqrt(sum_lanes(sums) / static_cast<float>(width) + 1e-6f);
          for (size_t j = 0; j < width; j += 4) {
              const auto z = vmulq_n_f32(vld1q_f32(residual + base + j), inv_rms);
              const auto gain = vaddq_f32(vdupq_n_f32(1.f), vld1q_f32(scale + j));
              vst1q_f32(hidden + base + j, vaddq_f32(vmulq_f32(z, gain), vld1q_f32(shift + j)));
          }
      }
    });
}

extern "C" void kit_swiglu(const float *packed, float *output, size_t rows, size_t width) {
    constexpr size_t block = 256;
    parallel_rows(rows, ^(size_t begin, size_t end) {
      alignas(64) float negative[block], exponential[block];
      for (size_t row = begin; row < end; ++row) {
          const float *gate = packed + row * 2 * width;
          const float *up = gate + width;
          for (size_t i = 0; i < width; i += block) {
              const int count = static_cast<int>(std::min(block, width - i));
              const int vector_end = count - count % 4;
              for (int j = 0; j < vector_end; j += 4)
                  vst1q_f32(negative + j, vnegq_f32(vld1q_f32(gate + i + j)));
              for (int j = vector_end; j < count; ++j)
                  negative[j] = -gate[i + j];
              vvexpf(exponential, negative, &count);
              for (int j = 0; j < vector_end; j += 4) {
                  const auto denominator = vaddq_f32(vdupq_n_f32(1.f), vld1q_f32(exponential + j));
                  const auto silu = vdivq_f32(vld1q_f32(gate + i + j), denominator);
                  vst1q_f32(output + row * width + i + j, vmulq_f32(silu, vld1q_f32(up + i + j)));
              }
              for (int j = vector_end; j < count; ++j) {
                  const float silu = gate[i + j] / (1.f + exponential[j]);
                  output[row * width + i + j] = silu * up[i + j];
              }
          }
      }
    });
}
