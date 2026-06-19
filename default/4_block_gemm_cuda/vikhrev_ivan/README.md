# Block Matrix Multiplication using CUDA

### Prerequisites

```
sudo apt install nvidia-cuda-toolkit
```

### Build

```
nvcc main.cu block_gemm_cuda.cu -o gemm
```


### Run
```
./block_gemm [number of elements]
```

### Results
Results for 1000 elements:
* Intel(R) Xeon(R) w5-3425, 12 cores,  3.20 GHz
* NVIDIA GeForce RTX 5090

```
 ./block_gemm 1000
Elements num (^2): 1000000
------------------------
Naive GEMM: 3871.28 ms
Block GEMM CUDA simple: 2.17694 ms
        Mean abs diff: 0.000182913974512
        Max diff: 0.001708984375
Block GEMM CUDA: 1.86082601547 ms
        Mean abs diff: 0.000182913974512
        Max diff: 0.001708984375
```
