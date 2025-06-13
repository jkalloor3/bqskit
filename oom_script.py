
from pyLIQTR.gate_decomp.gate_approximation import approximate_rz_direct
import numpy as np
import multiprocessing as mp

def worker():
    while True:
        rand_num = np.random.randint(4, 100000)
        rand_den = np.random.randint(rand_num // 2, 100000)
        rand_prec = np.random.randint(8, 20)
        out_str = approximate_rz_direct(rand_num, rand_den, 10)

if __name__ == '__main__':
    num_processes = mp.cpu_count()
    processes = []

    for _ in range(num_processes):
        p = mp.Process(target=worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes completed.")