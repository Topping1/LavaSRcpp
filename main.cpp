#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <string>
#include <memory>

const double PI = 3.14159265358979323846;

// ============================================================================
// CONFIGURATION (Hardcoded from config.yaml)
// ============================================================================
struct EnhancerConfig {
    int sample_rate = 44100;
    int n_fft = 2048;
    int hop_length = 512;
    int n_mels = 80;
    float f_min = 0.0f;
    float f_max = 8000.0f;
};

// ============================================================================
// DSP & MATH UTILITIES
// ============================================================================

// Compact Radix-2 Cooley-Tukey FFT
void fft(std::vector<std::complex<float>>& x, bool inverse) {
    int N = x.size();
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (int len = 2; len <= N; len <<= 1) {
        float angle = 2 * PI / len * (inverse ? 1 : -1);
        std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < N; i += len) {
            std::complex<float> w(1, 0);
            for (int j = 0; j < len / 2; j++) {
                std::complex<float> u = x[i + j];
                std::complex<float> v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < N; i++) x[i] /= N;
    }
}

// Lanczos Resampler (replaces scipy.signal.resample_poly)
std::vector<float> resample(const std::vector<float>& x, int sr_in, int sr_out) {
    if (sr_in == sr_out) return x;
    double ratio = (double)sr_out / sr_in;
    int out_len = std::round(x.size() * ratio);
    std::vector<float> y(out_len, 0.0f);
    int a = 5; // Lanczos window size
    double scale = std::min(1.0, ratio);
    
    for (int i = 0; i < out_len; i++) {
        double center = i / ratio;
        int left = std::max(0, (int)std::floor(center - a / scale));
        int right = std::min((int)x.size() - 1, (int)std::floor(center + a / scale));
        float sum = 0.0f, weight_sum = 0.0f;
        
        for (int j = left; j <= right; j++) {
            double x_val = (center - j) * scale;
            double weight = 1.0;
            if (x_val != 0.0) {
                double pi_x = PI * x_val;
                weight = std::sin(pi_x) * std::sin(pi_x / a) / (pi_x * pi_x / a);
            }
            sum += x[j] * weight;
            weight_sum += weight;
        }
        y[i] = sum / weight_sum;
    }
    return y;
}

std::vector<float> hann_periodic(int win_len) {
    std::vector<float> w(win_len);
    for (int i = 0; i < win_len; i++) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * PI * i / win_len));
    }
    return w;
}

std::vector<float> pad_reflect(const std::vector<float>& x, int pad_left, int pad_right) {
    int N = x.size();
    std::vector<float> y(N + pad_left + pad_right);
    for (int i = -pad_left; i < N + pad_right; i++) {
        int idx = i;
        while (idx < 0 || idx >= N) {
            if (idx < 0) idx = -idx;
            if (idx >= N) idx = 2 * N - 2 - idx;
        }
        y[i + pad_left] = x[idx];
    }
    return y;
}

// ============================================================================
// STFT / ISTFT
// ============================================================================

std::vector<std::vector<std::complex<float>>> stft(
    const std::vector<float>& x, int n_fft, int hop_length, int win_length, bool center_pad) 
{
    int pad = center_pad ? (n_fft / 2) : ((win_length - hop_length) / 2);
    std::vector<float> xpad = pad_reflect(x, pad, pad);
    if (xpad.size() < win_length) xpad.resize(win_length, 0.0f);

    std::vector<float> window = hann_periodic(win_length);
    int num_frames = (xpad.size() - win_length) / hop_length + 1;

    std::vector<std::vector<std::complex<float>>> spec(num_frames, std::vector<std::complex<float>>(n_fft / 2 + 1));

    for (int t = 0; t < num_frames; t++) {
        std::vector<std::complex<float>> frame(n_fft, {0, 0});
        for (int i = 0; i < win_length; i++) {
            frame[i] = {xpad[t * hop_length + i] * window[i], 0.0f};
        }
        fft(frame, false);
        for (int f = 0; f <= n_fft / 2; f++) {
            spec[t][f] = frame[f];
        }
    }
    return spec;
}

std::vector<float> istft(
    const std::vector<std::vector<std::complex<float>>>& spec,
    int n_fft, int hop_length, int win_length, bool center_pad, int target_len) 
{
    int pad = center_pad ? (n_fft / 2) : ((win_length - hop_length) / 2);
    std::vector<float> window = hann_periodic(win_length);

    int T = spec.size();
    int output_size = (T - 1) * hop_length + win_length;
    std::vector<float> y(output_size, 0.0f);
    std::vector<float> wenv(output_size, 0.0f);

    for (int t = 0; t < T; t++) {
        std::vector<std::complex<float>> frame(n_fft, {0, 0});
        for (int f = 0; f <= n_fft / 2; f++) {
            frame[f] = spec[t][f];
            if (f > 0 && f < n_fft / 2) {
                frame[n_fft - f] = std::conj(spec[t][f]);
            }
        }
        fft(frame, true);

        for (int i = 0; i < win_length; i++) {
            y[t * hop_length + i] += frame[i].real() * window[i];
            wenv[t * hop_length + i] += window[i] * window[i];
        }
    }

    std::vector<float> out;
    for (int i = pad; i < output_size - pad; i++) {
        out.push_back(y[i] / std::max(wenv[i], 1e-8f));
    }

    if (target_len > 0) {
        if (out.size() < target_len) out.resize(target_len, 0.0f);
        else out.resize(target_len);
    }
    return out;
}

// ============================================================================
// MEL FILTERBANK & FRONTEND
// ============================================================================

float hz_to_mel_slaney(float f) {
    if (f >= 1000.0f) return 15.0f + std::log(f / 1000.0f) / (std::log(6.4f) / 27.0f);
    return f / (200.0f / 3.0f);
}

float mel_to_hz_slaney(float m) {
    if (m >= 15.0f) return 1000.0f * std::exp((std::log(6.4f) / 27.0f) * (m - 15.0f));
    return (200.0f / 3.0f) * m;
}

std::vector<std::vector<float>> build_mel_filterbank_slaney(int sr, int n_fft, int n_mels, float fmin, float fmax) {
    int n_freqs = n_fft / 2 + 1;
    std::vector<float> fftfreqs(n_freqs);
    for (int i = 0; i < n_freqs; i++) fftfreqs[i] = (float)i * sr / n_fft;

    float m_min = hz_to_mel_slaney(fmin);
    float m_max = hz_to_mel_slaney(fmax);
    std::vector<float> f_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        f_pts[i] = mel_to_hz_slaney(m_min + i * (m_max - m_min) / (n_mels + 1));
    }

    std::vector<std::vector<float>> fb(n_mels, std::vector<float>(n_freqs, 0.0f));
    for (int i = 0; i < n_mels; i++) {
        float fdiff_left = std::max(f_pts[i+1] - f_pts[i], 1e-12f);
        float fdiff_right = std::max(f_pts[i+2] - f_pts[i+1], 1e-12f);
        float enorm = 2.0f / std::max(f_pts[i+2] - f_pts[i], 1e-12f);

        for (int j = 0; j < n_freqs; j++) {
            float lower = (fftfreqs[j] - f_pts[i]) / fdiff_left;
            float upper = (f_pts[i+2] - fftfreqs[j]) / fdiff_right;
            fb[i][j] = std::max(0.0f, std::min(lower, upper)) * enorm;
        }
    }
    return fb;
}

std::vector<std::vector<float>> mel_frontend(const std::vector<float>& wav, int sr, int n_fft, int hop_length, int n_mels, float fmin, float fmax) {
    auto spec = stft(wav, n_fft, hop_length, n_fft, false);
    auto fb = build_mel_filterbank_slaney(sr, n_fft, n_mels, fmin, fmax);

    int T = spec.size();
    std::vector<std::vector<float>> mel(n_mels, std::vector<float>(T, 0.0f));

    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int f = 0; f <= n_fft / 2; f++) {
                sum += fb[m][f] * std::abs(spec[t][f]);
            }
            mel[m][t] = std::log(std::max(sum, 1e-5f));
        }
    }
    return mel;
}

// ============================================================================
// FAST LR MERGE
// ============================================================================

std::vector<float> fast_lr_merge(const std::vector<float>& enhanced, const std::vector<float>& original, int sr = 48000, int cutoff = 4000, int transition_bins = 256) {
    int N = enhanced.size();
    int N_pow2 = 1;
    while (N_pow2 < N) N_pow2 <<= 1;

    std::vector<std::complex<float>> spec1(N_pow2, {0, 0});
    std::vector<std::complex<float>> spec2(N_pow2, {0, 0});
    for (int i = 0; i < N; i++) {
        spec1[i] = {enhanced[i], 0};
        spec2[i] = {original[i], 0};
    }

    fft(spec1, false);
    fft(spec2, false);

    int n_bins = N_pow2 / 2 + 1;
    int cutoff_bin = (int)((cutoff / (sr / 2.0f)) * n_bins);
    int half = transition_bins / 2;
    int start = std::max(0, cutoff_bin - half);
    int end = std::min(n_bins, cutoff_bin + half);

    std::vector<float> mask(n_bins, 1.0f);
    for (int i = 0; i < start; i++) mask[i] = 0.0f;
    if (end > start) {
        for (int i = start; i < end; i++) {
            float x = -1.0f + 2.0f * (i - start) / (end - start - 1);
            float t = (x + 1.0f) / 2.0f;
            mask[i] = 3 * t * t - 2 * t * t * t;
        }
    }

    for (int i = 0; i < n_bins; i++) {
        spec2[i] = spec2[i] + (spec1[i] - spec2[i]) * mask[i];
        if (i > 0 && i < N_pow2 / 2) {
            spec2[N_pow2 - i] = std::conj(spec2[i]);
        }
    }
    spec2[N_pow2 / 2] = {spec2[N_pow2 / 2].real(), 0};

    fft(spec2, true);

    std::vector<float> out(N);
    for (int i = 0; i < N; i++) out[i] = spec2[i].real();
    return out;
}

// ============================================================================
// LAVA DENOISER
// ============================================================================

class LavaDenoiser {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "LavaDenoiser"};
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name, output_name;

    int n_fft = 512, hop_len = 256, win_len = 512;
    int chunk_frames = 63, chunk_hop_frames = 21;
    std::vector<float> chunk_weight;

public:
    LavaDenoiser(const std::string& model_path, int intra_threads, int inter_threads) {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(intra_threads);
        so.SetInterOpNumThreads(inter_threads);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), so);
        input_name = session->GetInputNameAllocated(0, allocator).get();
        output_name = session->GetOutputNameAllocated(0, allocator).get();

        chunk_weight.resize(chunk_frames);
        for (int i = 0; i < chunk_frames; i++) {
            float w = 0.5f * (1.0f - std::cos(2.0f * PI * i / (chunk_frames - 1)));
            chunk_weight[i] = std::max(w * w, 1e-4f);
        }
    }

    std::vector<float> infer(const std::vector<float>& wav) {
        auto spec = stft(wav, n_fft, hop_len, win_len, true);
        int T = spec.size();
        int F = n_fft / 2 + 1;

        std::vector<float> flat_spec(2 * T * F);
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                flat_spec[0 * T * F + t * F + f] = spec[t][f].real();
                flat_spec[1 * T * F + t * F + f] = spec[t][f].imag();
            }
        }

        int L = chunk_frames;
        int H = chunk_hop_frames;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        if (T <= L) {
            std::vector<float> chunk(2 * L * F, 0.0f);
            for (int c = 0; c < 2; c++)
                for (int t = 0; t < T; t++)
                    for (int f = 0; f < F; f++)
                        chunk[c * L * F + t * F + f] = flat_spec[c * T * F + t * F + f];

            std::vector<int64_t> shape = {1, 2, L, F};
            Ort::Value in_tensor = Ort::Value::CreateTensor<float>(mem_info, chunk.data(), chunk.size(), shape.data(), shape.size());
            const char* in_names[] = {input_name.c_str()};
            const char* out_names[] = {output_name.c_str()};
            auto out_tensors = session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
            float* out_ptr = out_tensors[0].GetTensorMutableData<float>();

            for (int c = 0; c < 2; c++)
                for (int t = 0; t < T; t++)
                    for (int f = 0; f < F; f++)
                        flat_spec[c * T * F + t * F + f] = out_ptr[c * L * F + t * F + f];
        } else {
            std::vector<int> starts;
            for (int s = 0; s <= T - L; s += H) starts.push_back(s);
            if (starts.back() != T - L) starts.push_back(T - L);

            std::vector<float> acc(2 * T * F, 0.0f);
            std::vector<float> wacc(T, 0.0f);

            for (int start : starts) {
                std::vector<float> chunk(2 * L * F, 0.0f);
                for (int c = 0; c < 2; c++)
                    for (int t = 0; t < L; t++)
                        for (int f = 0; f < F; f++)
                            chunk[c * L * F + t * F + f] = flat_spec[c * T * F + (start + t) * F + f];

                std::vector<int64_t> shape = {1, 2, L, F};
                Ort::Value in_tensor = Ort::Value::CreateTensor<float>(mem_info, chunk.data(), chunk.size(), shape.data(), shape.size());
                const char* in_names[] = {input_name.c_str()};
                const char* out_names[] = {output_name.c_str()};
                auto out_tensors = session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
                float* out_ptr = out_tensors[0].GetTensorMutableData<float>();

                for (int c = 0; c < 2; c++)
                    for (int t = 0; t < L; t++)
                        for (int f = 0; f < F; f++)
                            acc[c * T * F + (start + t) * F + f] += out_ptr[c * L * F + t * F + f] * chunk_weight[t];

                for (int t = 0; t < L; t++) wacc[start + t] += chunk_weight[t];
            }

            for (int c = 0; c < 2; c++)
                for (int t = 0; t < T; t++) {
                    float w = std::max(wacc[t], 1e-6f);
                    for (int f = 0; f < F; f++)
                        flat_spec[c * T * F + t * F + f] = acc[c * T * F + t * F + f] / w;
                }
        }

        std::vector<std::vector<std::complex<float>>> spec_enh(T, std::vector<std::complex<float>>(F));
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                spec_enh[t][f] = {flat_spec[0 * T * F + t * F + f], flat_spec[1 * T * F + t * F + f]};
            }
        }

        return istft(spec_enh, n_fft, hop_len, win_len, true, wav.size());
    }
};

// ============================================================================
// LAVA ENHANCER
// ============================================================================

class LavaEnhancer {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "LavaEnhancer"};
    std::unique_ptr<Ort::Session> backbone_session;
    std::unique_ptr<Ort::Session> spec_head_session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string bb_in, bb_out, sh_in, sh_out1, sh_out2;
    EnhancerConfig cfg;

public:
    LavaEnhancer(const std::string& bb_path, const std::string& sh_path, int intra_threads, int inter_threads) {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(intra_threads);
        so.SetInterOpNumThreads(inter_threads);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        backbone_session = std::make_unique<Ort::Session>(env, bb_path.c_str(), so);
        spec_head_session = std::make_unique<Ort::Session>(env, sh_path.c_str(), so);

        bb_in = backbone_session->GetInputNameAllocated(0, allocator).get();
        bb_out = backbone_session->GetOutputNameAllocated(0, allocator).get();
        sh_in = spec_head_session->GetInputNameAllocated(0, allocator).get();
        sh_out1 = spec_head_session->GetOutputNameAllocated(0, allocator).get();
        sh_out2 = spec_head_session->GetOutputNameAllocated(1, allocator).get();
    }

    std::vector<float> infer(const std::vector<float>& wav) {
        auto mel = mel_frontend(wav, cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels, cfg.f_min, cfg.f_max);
        int T = mel[0].size();

        std::vector<float> flat_mel(cfg.n_mels * T);
        for (int m = 0; m < cfg.n_mels; m++)
            for (int t = 0; t < T; t++)
                flat_mel[m * T + t] = mel[m][t];

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Backbone
        std::vector<int64_t> bb_shape = {1, cfg.n_mels, T};
        Ort::Value bb_tensor = Ort::Value::CreateTensor<float>(mem_info, flat_mel.data(), flat_mel.size(), bb_shape.data(), bb_shape.size());
        const char* bb_in_names[] = {bb_in.c_str()};
        const char* bb_out_names[] = {bb_out.c_str()};
        auto bb_out_tensors = backbone_session->Run(Ort::RunOptions{nullptr}, bb_in_names, &bb_tensor, 1, bb_out_names, 1);
        
        // SpecHead
        const char* sh_in_names[] = {sh_in.c_str()};
        const char* sh_out_names[] = {sh_out1.c_str(), sh_out2.c_str()};
        auto sh_out_tensors = spec_head_session->Run(Ort::RunOptions{nullptr}, sh_in_names, &bb_out_tensors[0], 1, sh_out_names, 2);
        
        float* real_ptr = sh_out_tensors[0].GetTensorMutableData<float>();
        float* imag_ptr = sh_out_tensors[1].GetTensorMutableData<float>();
        int F = cfg.n_fft / 2 + 1;

        std::vector<std::vector<std::complex<float>>> spec(T, std::vector<std::complex<float>>(F));
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                spec[t][f] = {real_ptr[f * T + t], imag_ptr[f * T + t]};
            }
        }

        auto enhanced = istft(spec, cfg.n_fft, cfg.hop_length, cfg.n_fft, false, wav.size());
        return fast_lr_merge(enhanced, wav, 48000);
    }
};

// ============================================================================
// MAIN ENGINE & CLI
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.wav> [-o output.wav] [--denoise][--denoiser-onnx path] [--enhancer-backbone-onnx path][--enhancer-spec-head-onnx path]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = "output.wav";
    bool apply_denoise = false;
    std::string denoiser_onnx = "denoiser_core_legacy_fixed63.onnx";
    std::string enhancer_bb_onnx = "enhancer_backbone.onnx";
    std::string enhancer_sh_onnx = "enhancer_spec_head.onnx";

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--output") output_path = argv[++i];
        else if (arg == "--denoise") apply_denoise = true;
        else if (arg == "--denoiser-onnx") denoiser_onnx = argv[++i];
        else if (arg == "--enhancer-backbone-onnx") enhancer_bb_onnx = argv[++i];
        else if (arg == "--enhancer-spec-head-onnx") enhancer_sh_onnx = argv[++i];
    }

    std::cout << "Loading audio..." << std::endl;
    drwav wav;
    if (!drwav_init_file(&wav, input_path.c_str(), nullptr)) {
        std::cerr << "Failed to open " << input_path << std::endl;
        return 1;
    }

    std::vector<float> audio(wav.totalPCMFrameCount * wav.channels);
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, audio.data());
    
    std::vector<float> mono(wav.totalPCMFrameCount);
    for (size_t i = 0; i < wav.totalPCMFrameCount; i++) {
        float sum = 0;
        for (int c = 0; c < wav.channels; c++) sum += audio[i * wav.channels + c];
        mono[i] = sum / wav.channels;
    }
    drwav_uninit(&wav);

    std::vector<float> wav_16k = resample(mono, wav.sampleRate, 16000);

    if (apply_denoise) {
        std::cout << "Running denoiser..." << std::endl;
        LavaDenoiser denoiser(denoiser_onnx, 1, 1);
        wav_16k = denoiser.infer(wav_16k);
    }

    std::cout << "Running enhancer..." << std::endl;
    std::vector<float> wav_48k = resample(wav_16k, 16000, 48000);
    LavaEnhancer enhancer(enhancer_bb_onnx, enhancer_sh_onnx, 1, 1);
    std::vector<float> enhanced = enhancer.infer(wav_48k);

    std::cout << "Saving: " << output_path << std::endl;
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = 48000;
    format.bitsPerSample = 32;

    drwav wav_out;
    if (!drwav_init_file_write(&wav_out, output_path.c_str(), &format, nullptr)) {
        std::cerr << "Failed to open " << output_path << " for writing." << std::endl;
        return 1;
    }
    drwav_write_pcm_frames(&wav_out, enhanced.size(), enhanced.data());
    drwav_uninit(&wav_out);

    std::cout << "Done!" << std::endl;
    return 0;
}