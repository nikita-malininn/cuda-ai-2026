# GELU CUDA

### Prerequisites

```
sudo apt install nvidia-cuda-toolkit
```

### Build

```
nvcc main.cu gelu_cuda.cu -o gelu
```


### Run
```
./gelu [number of elements]
```

### Results
Results for 134217728 elements:
* Intel(R) Xeon(R) w5-3425, 12 cores,  3.20 GHz
* NVIDIA GeForce RTX 5090

```
Elements num: 134217728
------------------------
Gelu: 3759.87 ms
Gelu CUDA Naive: 444.94 ms
        Mean abs diff: 2.08636957665e-08
        Max diff: 4.76837158203e-07
Gelu CUDA: 444.945068359 ms
        Mean abs diff: 2.08636957665e-08
        Max diff: 4.76837158203e-07
```