# Naive Matrix Multiplication using CUDA

### Prerequisites

```
sudo apt install nvidia-cuda-toolkit
```

### Build

```
nvcc main.cu naive_gemm_cuda.cu -o gemm
```


### Run
```
./gemm [number of elements]
```

### Results
Results for 1000 elements:
* Intel(R) Xeon(R) w5-3425, 12 cores,  3.20 GHz
* NVIDIA GeForce RTX 5090

```
 ./gemm 1000
Elements num (^2): 1000000
------------------------
Naive GEMM: 3874.55 ms
Naive GEMM CUDA simple: 2.77034 ms
        Mean abs diff: 0.000182986125465
        Max diff: 0.001953125
Naive GEMM CUDA: 2.47920918465 ms
        Mean abs diff: 0.000182986125465
        Max diff: 0.001953125
```