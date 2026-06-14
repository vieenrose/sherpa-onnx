#!/bin/bash
# Build sherpa-onnx against the cuDNN-free onnxruntime 1.11 CUDA EP for the
# NVIDIA Jetson Nano gen1 (sm_53 Maxwell, JetPack 4.6.x / CUDA 10.2). Run inside
# an aarch64 L4T r32.7 / CUDA-10.2 container (also reproduces on a GB10 via PTX-JIT).
#
# Why: onnxruntime's stock CUDA EP pulls in cuDNN (~782 MB resident on cuDNN 8) —
# prohibitive on a 4 GB Nano. The cuDNN-free EP (cuBLAS-only) comes from
#   github.com/vieenrose/onnxruntime @ cudnn-free-cuda-jetson-nano-gen1  (ORT 1.11.0).
#
# Prereq: package that ORT build into the tarball sherpa expects, e.g.
#   stage=/tmp/ortpkg; mkdir -p $stage/{include,lib}
#   cp <ort>/include/onnxruntime/core/session/onnxruntime_{c,cxx}_api.h \
#      <ort>/include/onnxruntime/core/session/onnxruntime_cxx_inline.h \
#      <ort>/include/onnxruntime/core/session/onnxruntime_*_config_keys.h \
#      <ort>/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h $stage/include/
#   cp -P <ort>/build/.../libonnxruntime.so* <ort>/build/.../libonnxruntime_providers_*.so $stage/lib/
#   (cd $stage && tar cjf /tmp/onnxruntime-linux-aarch64-gpu-1.11.0.tar.bz2 include lib)
set -e
command -v g++-8 >/dev/null || { apt-get update; apt-get install -y --no-install-recommends g++-8 gcc-8; }
cd "$(dirname "$0")/.."
rm -rf build-nano-ort && mkdir build-nano-ort && cd build-nano-ort
# sherpa's cmake finds /tmp/onnxruntime-linux-aarch64-gpu-1.11.0.tar.bz2 via possible_file_locations;
# URL_HASH is disabled (see cmake/onnxruntime-linux-aarch64-gpu.cmake) so the local cuDNN-free tarball is used.
cmake -DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_GPU=ON \
  -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.11.0 \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_TESTS=OFF -DSHERPA_ONNX_ENABLE_C_API=OFF \
  -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 ..
make -j"$(nproc)" sherpa-onnx-offline sherpa-onnx-offline-tts
echo "BUILT: build-nano-ort/bin/{sherpa-onnx-offline,sherpa-onnx-offline-tts}"
echo
echo "RUNTIME NOTE: the cuDNN-free EP has no cuDNN-only FusedConv kernel, so disable"
echo "the Conv+activation fusion via a provider config (ORT_ENABLE_BASIC=1):"
echo '  echo "GraphOptimizationLevel=1" > /tmp/ortcfg.txt'
echo '  sherpa-onnx-offline --provider=cuda:/tmp/ortcfg.txt ...    # (and -offline-tts)'
echo "Also use opset<=16 models on ORT 1.11 (melo8k: decompose LayerNormalization to opset 16)."
