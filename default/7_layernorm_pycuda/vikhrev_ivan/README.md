# Layer Norm Implementation in PyCUDA

## Prerequisites

- Python 3.12 or higher


## Usage

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py -n <number_of_elements_in_a_row>
```

## Results
Results for 1000000000000 elements:
* Intel(R) Xeon(R) w5-3425, 12 cores,  3.20 GHz
* NVIDIA GeForce RTX 5090

```
GPU: NVIDIA GeForce RTX 5090
Compute capability: (12, 0)
Elements num (^2): 1000000000000
NumPy LayerNorm:
        mean time: 58.6111 ms
        min time:  55.8111 ms
        max time:  62.3210 ms
PyCUDA LayerNorm:
        mean time: 49.0088 ms
        min time:  47.4061 ms
        max time:  50.4117 ms
Difference:
        mean abs diff: 0.00000388
        max abs diff:  0.00004578
        allclose:      True
```