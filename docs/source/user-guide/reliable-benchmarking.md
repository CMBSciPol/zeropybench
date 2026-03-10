# Reliable Benchmarking

Benchmark results can vary significantly due to system-level factors. This guide covers best practices for obtaining reproducible and reliable measurements.

## What ZeroPyBench Does Automatically

zeropybench already implements several best practices:

- **Multiple repetitions with median**: Reduces the impact of outliers
- **Auto-scaling**: Automatically determines the number of iterations for reliable measurements
- **JAX compilation separation**: Reports compilation time separately from execution time
- **Proper synchronization**: Uses `block_until_ready()` for accurate JAX timing

## CPU Benchmarking

### Disable Frequency Scaling

Modern CPUs dynamically adjust their frequency based on load and temperature. This can cause significant variance in benchmark results.

```bash
# Set the CPU governor to performance mode (requires root)
sudo cpupower frequency-set -g performance

# Verify the setting
cpupower frequency-info
```

To revert to the default:
```bash
sudo cpupower frequency-set -g powersave  # or ondemand
```

### Disable Turbo Boost

Turbo boost can cause inconsistent results as the CPU may throttle under sustained load.

**Intel CPUs:**
```bash
# Disable turbo boost
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Re-enable turbo boost
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

**AMD CPUs:**
```bash
# Disable turbo boost
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost

# Re-enable turbo boost
echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

### CPU Isolation

Isolate CPU cores to prevent the OS scheduler from interrupting your benchmark.

**Runtime isolation with taskset:**
```bash
# Run on cores 0-3 only
taskset -c 0-3 python benchmark.py
```

**Boot-time isolation (more effective):**

Add to your kernel boot parameters in `/etc/default/grub`:
```
GRUB_CMDLINE_LINUX="isolcpus=0-3 nohz_full=0-3"
```

Then update GRUB and reboot:
```bash
sudo update-grub
sudo reboot
```

### Process Priority

Increase the priority of your benchmark process:

```bash
# Run with highest priority (requires root)
sudo nice -n -20 python benchmark.py

# Or with real-time scheduling
sudo chrt -f 99 python benchmark.py
```

### Disable Hyperthreading

Hyperthreading can introduce variability. Disable it in BIOS or at runtime:

```bash
# Disable hyperthreading (example for 8 physical cores with HT)
echo 0 | sudo tee /sys/devices/system/cpu/cpu{8..15}/online
```

## GPU Benchmarking (NVIDIA)

### Enable Persistence Mode

Keeps the GPU initialized between runs, reducing startup overhead:

```bash
sudo nvidia-smi -pm 1
```

### Lock GPU Clocks

Prevent dynamic frequency scaling on the GPU:

```bash
# Query supported clocks
nvidia-smi -q -d SUPPORTED_CLOCKS

# Lock graphics clocks (example: 1500 MHz)
sudo nvidia-smi -lgc 1500,1500

# Lock memory clocks (example: 5001 MHz)
sudo nvidia-smi -lmc 5001

# Reset to default
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

### Exclusive Process Mode

Ensure only one process can use the GPU:

```bash
# Set exclusive process mode
sudo nvidia-smi -c EXCLUSIVE_PROCESS

# Reset to default (shared mode)
sudo nvidia-smi -c DEFAULT
```

### Disable ECC Memory (Optional)

On GPUs with ECC memory, disabling it can provide ~10% more memory bandwidth. This is a persistent setting that requires a reboot:

```bash
# Check current ECC status
nvidia-smi -q | grep -i ecc

# Disable ECC (requires reboot)
sudo nvidia-smi -e 0
```

### Monitor GPU State

Before running benchmarks, verify GPU state:

```bash
# Check temperatures, clocks, and utilization
nvidia-smi -q -d PERFORMANCE

# Monitor in real-time
watch -n 1 nvidia-smi
```

## Environment Variables

### JAX-specific

```bash
# Disable JAX memory preallocation (useful for memory profiling)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0

# Disable JAX compilation cache (for cold-start benchmarks)
export JAX_ENABLE_COMPILATION_CACHE=false
```

### General Python

```bash
# Disable Python's hash randomization for reproducibility
export PYTHONHASHSEED=0
```

## Quick Setup Script

Here's a script that applies common optimizations:

```bash
#!/bin/bash
# setup_benchmark_env.sh - Run as root

set -e

echo "Setting up benchmark environment..."

# CPU optimizations
cpupower frequency-set -g performance
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || \
echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || \
echo "Could not disable turbo boost"

# GPU optimizations (if NVIDIA GPU present)
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -pm 1
    # Optionally lock clocks here
fi

echo "Benchmark environment ready."
echo "Run your benchmark with: taskset -c 0-3 nice -n -20 python benchmark.py"
```

## Verification Checklist

Before running benchmarks, verify:

- [ ] CPU governor is set to `performance`
- [ ] Turbo boost is disabled
- [ ] GPU clocks are locked (for GPU benchmarks)
- [ ] No other intensive processes are running
- [ ] System temperature is stable
- [ ] Sufficient warm-up iterations have been run

## Interpreting Results

Even with all optimizations, some variance is expected:

- **< 1% variance**: Excellent, highly reproducible
- **1-5% variance**: Good, typical for well-controlled environments
- **5-10% variance**: Acceptable, may indicate some system noise
- **> 10% variance**: Investigate system configuration

zeropybench reports the interquartile range (IQR) as a percentage, which helps identify unstable measurements.
