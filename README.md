# LavaSRcpp

C++ runtime for LavaSR using ONNX Runtime.

This implementation runs the LavaSR pipeline in C++ with ONNX Runtime and custom DSP utilities implemented in `main.cpp`. The current pipeline supports:

- mono WAV input
- optional denoising with the ULUNAS ONNX denoiser
- 16 kHz to 48 kHz resampling
- enhancement with the ONNX backbone and ONNX spectrogram head
- WAV output at 48 kHz

The program uses three ONNX models at runtime:

- `denoiser_core_legacy_fixed63.onnx`
- `enhancer_backbone.onnx` (and `enhancer_backbone.onnx.data`)
- `enhancer_spec_head.onnx` (and `enhancer_spec_head.onnx.data`)

The ONNX models are located [here](https://github.com/Topping1/LavaSRcpp/releases/tag/Alpha-v.01)

The app reads and writes WAV files with `dr_wav.h`, which is an external single-file audio library from the `dr_libs` project by David Reid. Include the license and attribution required by that project when redistributing the repository.

## Requirements

Before compiling, install **ONNX Runtime** and make sure its CMake package is discoverable by `find_package(onnxruntime REQUIRED)`.

You also need:

- CMake 3.14+
- a C++17 compiler
- the ONNX model files from the release page
- `dr_wav.h` present in the repository root next to `main.cpp`

This project’s `CMakeLists.txt` expects ONNX Runtime to be installed already and links against `onnxruntime::onnxruntime`.

## Model files

Download the required ONNX files from the release page:

- Release page: https://github.com/Topping1/LavaSRcpp/releases/tag/Alpha-v.01

Place the ONNX files and any matching `.onnx.data` sidecar files in the same directory as the executable, or pass explicit paths through the CLI arguments.

Expected runtime model files:

- `denoiser_core_legacy_fixed63.onnx`
- `enhancer_backbone.onnx`
- `enhancer_backbone.onnx.data`
- `enhancer_spec_head.onnx`
- `enhancer_spec_head.onnx.data`

## Expected repository structure

A minimal layout is:

```text
LavaSRcpp/
├── CMakeLists.txt
├── main.cpp
├── dr_wav.h
├── denoiser_core_legacy_fixed63.onnx
├── enhancer_backbone.onnx
├── enhancer_backbone.onnx.data
├── enhancer_spec_head.onnx
├── enhancer_spec_head.onnx.data
└── build/
```

If you keep the ONNX files elsewhere, pass their paths with the CLI flags.

## Build

Example build steps:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

If CMake cannot find ONNX Runtime automatically, point it to your ONNX Runtime installation using the appropriate CMake variable or prefix path for your platform.

## Run

Basic usage:

```bash
./lavasr input.wav
```

Write to a specific output file:

```bash
./lavasr input.wav -o output.wav
```

Enable denoising:

```bash
./lavasr input.wav -o output.wav --denoise
```

Pass explicit model paths:

```bash
./lavasr input.wav \
  --denoise \
  --denoiser-onnx /path/to/denoiser_core_legacy_fixed63.onnx \
  --enhancer-backbone-onnx /path/to/enhancer_backbone.onnx \
  --enhancer-spec-head-onnx /path/to/enhancer_spec_head.onnx \
  -o output.wav
```

## Notes

- Input is read as WAV and mixed to mono if needed.
- Audio is resampled internally to 16 kHz for denoising and 48 kHz for enhancement.
- Output is written as 32-bit float mono WAV at 48 kHz.
- The DSP pipeline in `main.cpp` includes custom FFT, STFT/ISTFT, mel frontend, resampling, and final spectral merge.
