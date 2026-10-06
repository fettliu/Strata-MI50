#!/bin/sh
# gfx906 wave32 flag-spelling sweep. Usage: gfx906_wave32_probe.sh [rocm-root]
# rocm-root defaults to /opt/rocm; on a TheRock-wheels build pass the `rocm-sdk path --root` output.
# Compiles tools/hip/gfx906_wave32_probe.hip with each candidate spelling of the wave32 request and runs it;
# the first "WAVE32 OK" verdict tells which spelling to put in STRATA_HIP_WAVE32_FLAGS (cmake/hip_backend.cmake).
set -u
ROOT="${1:-/opt/rocm}"
HIPCC="$ROOT/bin/hipcc"
HERE=$(dirname "$0")
SRC="$HERE/gfx906_wave32_probe.hip"
OUT_DIR=$(mktemp -d)
[ -x "$HIPCC" ] || { echo "hipcc not found at $HIPCC"; exit 1; }
for flags in "" "-mno-wavefrontsize64" "-Xclang -target-feature -Xclang +wavefrontsize32" "-Xclang -target-feature -Xclang -wavefrontsize64"; do
    echo "=== flags: [$flags]"
    # shellcheck disable=SC2086
    if ! "$HIPCC" --offload-arch=gfx906 $flags "$SRC" -o "$OUT_DIR/probe" 2>"$OUT_DIR/err"; then
        echo "  compile failed:"; head -3 "$OUT_DIR/err"; continue
    fi
    if "$OUT_DIR/probe"; then :; else echo "  (probe says NOT wave32)"; fi
    echo
done
rm -rf "$OUT_DIR"
