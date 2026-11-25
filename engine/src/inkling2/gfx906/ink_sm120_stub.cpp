// engine/src/inkling2/gfx906/ink_sm120_stub.cpp - link-time stand-ins for the sm_120a (Blackwell) MoE entry points.
//
// The CUDA backend (compiled here as HIP for gfx906) calls strata::ink::sm120::* from launch_ink_moe_dispatch_only,
// but those symbols live in ink_moe_fp4_sm120.cu, a TU built only with -gencode arch=compute_120a,code=sm_120a
// (Blackwell tensor cores).  A gfx906 build has no such TU, so the link would fail on 7 undefined symbols.
//
// Every one of those calls is behind `sm120::available()` at run time (ink_moe_cuda.cuh:3678,
// `const bool use_sm120_tc = (!force_exact_dec && M >= 24 && sm120::available())`).  available() here returns false
// unconditionally - gfx906 is wave64 GCN, not Blackwell - so the other six are unreachable.  They abort() rather
// than returning silently: if control ever reaches one, an unavailable path is being taken and the want is to know,
// not to compute garbage.
#include <cstdio>
#include <cstdlib>
#include "ink_moe_fp4_sm120.cuh"

namespace strata::ink {
namespace sm120 {

bool available() { return false; }

static void unavailable(const char* fn) {
    std::fprintf(stderr, "strata::ink::sm120::%s is a gfx906 link-time stub and must not be called\n", fn);
    std::abort();
}

void quantize_act_fp4(const float*, int, int, uint8_t*, cudaStream_t) { unavailable("quantize_act_fp4"); }

void quantize_act_mx8(const float*, int, int, uint8_t*, uint8_t*, cudaStream_t) { unavailable("quantize_act_mx8"); }

void moe_sf4_gu_tc(const uint8_t*, const InkExpertDev*, InkMoeTables, const uint8_t*, uint8_t*, int, int, cudaStream_t) {
    unavailable("moe_sf4_gu_tc");
}

void moe_sf4_dn_tc(const uint8_t*, const InkExpertDev*, InkMoeTables, const uint8_t*, const float*, float*, int, int,
                   cudaStream_t) {
    unavailable("moe_sf4_dn_tc");
}

void moe_mx8_gu_tc(const uint8_t*, const InkExpertDev*, InkMoeTables, const uint8_t*, const uint8_t*, uint8_t*,
                   uint8_t*, int, int, cudaStream_t) {
    unavailable("moe_mx8_gu_tc");
}

void moe_mx8_dn_tc(const uint8_t*, const InkExpertDev*, InkMoeTables, const uint8_t*, const uint8_t*, const float*,
                   float*, int, int, cudaStream_t) {
    unavailable("moe_mx8_dn_tc");
}

}  // namespace sm120
}  // namespace strata::ink
