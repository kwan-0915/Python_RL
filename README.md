# About
This project borrowed the concept and implemented the portfolio 
re-balancing model from [1].

The fundamental model in here is D3PG from [2], the architecture of the D3PG model can be found
in ```architecture``` folder.

The system was tested under Windows 10 Pro Version 21H2 (OS Build 19044.1586), Intel Core i7-10750H, Nvidia GTX 1650 Ti.

CUDA version : 
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Thu_Feb_10_19:03:51_Pacific_Standard_Time_2022
Cuda compilation tools, release 11.6, V11.6.112
Build cuda_11.6.r11.6/compiler.30978841_0
```

# Usage 
* Train
```
python d3pg_main.py 
--mode "train"
--config_file "YOUR_PATH_TO_fx_d3pg.yml"
--data_file "YOUR_PATH_TO_fx_data"
--num_cpus NUM_OF_CPUS
--num_gpus NUM_OF_GPUS
--result_file "YOUR_PATH_TO_results_XXXX"
```

* Evaluate
```
python d3pg_main.py 
--mode "eval"
--config_file "YOUR_PATH_TO_fx_d3pg.yml"
--data_file "YOUR_PATH_TO_fx_data"
--num_cpus NUM_OF_CPUS
--num_gpus NUM_OF_GPUS
--result_file "YOUR_PATH_TO_results_XXXX"
```

# Special Remarks
Since the system was developed under Windows, it might be not runnable
in Linux. For example, in ```pytorch```, no processes can share memory
to other processes, but in Linux it will not be a problem. That's why in
this project ```pytorch multiprocess``` is not used.

However, as the name suggested D3PG was a DISTRIBUTED model. In order to solve this issues, ```ray``` was used. It is a distributed system for
python, you can visit (https://www.ray.io/) for more information.

# Dependency
All the libraries and corresponding version are listed under ```requirements.txt```

# Reference
1. A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem, (https://arxiv.org/abs/1706.10059)
2. Distributed Distributional Deterministic Policy Gradients, (https://arxiv.org/abs/1804.08617)