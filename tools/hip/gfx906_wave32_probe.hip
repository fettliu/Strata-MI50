// gfx906 wave32 probe for the Strata HIP fork.
//
// Strata's HIP kernels assume wave32: full-wave shuffle masks are 0xffffffff and ballots are cast to 32-bit words
// (see include/strata/hip_compat/intrinsics.hpp). Vega / CDNA1 cards default to wave64 code objects, so the fork
// builds must ask clang for wave32 (amdhsa_wavefront_size32) and the driver must actually run such a queue on the
// card. Compile this file with a candidate flag spelling and run it on the card; the verdict lines below answer
// both questions in one go. See gfx906_wave32_probe.sh for the sweep of spellings.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void wave_probe(int* out) {
    out[0] = warpSize;  // compile/runtime-visible wave width for this code object
    if (threadIdx.x == 0) {
        // lane 0 must read lane 1's value: holds on wave32 and on wave64 alike, so this only fails loudly below.
        out[1] = __shfl_xor(1000 + static_cast<int>(threadIdx.x), 1);
    }
    // The decisive test: with 64 threads and the predicate `threadIdx.x < 32`, a wave32 queue has every wave see
    // all 32 lanes set (0xffffffff); a wave64 queue has the single wave set bits 32..63 as well.
    const unsigned long long b = __ballot(threadIdx.x < 32);
    out[2] = (b == 0xffffffffull) ? 32 : (b == 0xffffffffffffffffull ? 64 : 0);
    // Pairing info only: lane 33 xor 1 is lane 32 on wave32 (second wave) and on wave64 alike, so this value does
    // not discriminate; it is recorded to catch misbehaving shuffles, not to pick the wave width.
    if (threadIdx.x == 33)
        out[3] = __shfl_xor(1000 + static_cast<int>(threadIdx.x), 1);
}

int main() {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) {
        std::printf("no HIP device\n");
        return 1;
    }
    std::printf("device=%s gcnArchName=%s property.warpSize=%d\n", p.name, p.gcnArchName, p.warpSize);
    int* d = nullptr;
    if (hipMalloc(&d, 4 * sizeof(int)) != hipSuccess) {
        std::printf("hipMalloc failed\n");
        return 1;
    }
    wave_probe<<<1, 64>>>(d);
    const hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        std::printf("launch failed: %s\n", hipGetErrorString(e));
        return 1;
    }
    int h[4] = {0, 0, 0, 0};
    hipMemcpy(h, d, 4 * sizeof(int), hipMemcpyDeviceToHost);
    std::printf("kernel warpSize=%d lane0shuffle=%d ballot_wave=%d lane33shuffle=%d\n", h[0], h[1], h[2], h[3]);
    const bool wave32 = h[0] == 32 && h[2] == 32;
    std::printf("VERDICT: %s\n", wave32 ? "WAVE32 OK" : "NOT wave32 (unsuitable for Strata)");
    return wave32 ? 0 : 2;
}
