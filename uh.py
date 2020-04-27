import numpy as np
import time
import cupy as cp

### Numpy and CPU
s = time.time()
x_cpu = np.ones((1000,1000,1))
e = time.time()
c = (e - s)
### CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000,1000,1))
cp.cuda.Stream.null.synchronize()
e = time.time()
g = (e - s)


print ("ran", g/c, "times faster")