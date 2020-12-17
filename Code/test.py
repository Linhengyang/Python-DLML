from d2l import torch as d2l
import math
import torch
import numpy as np
import time

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()
    
    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

n = 10000
a = torch.ones(n)
b = torch.ones(n)
c = torch.zeros(n)

# Method 1: for loop
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')


# Method 2: reloaded + operator
timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')