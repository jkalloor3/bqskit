import time
import os
import glob
from os.path import join

sleep_time = 0.05

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats"
    file = "get_superensemble"
    # file = "full_compile"
    circs = ["Heisenberg_7", "TFXY_8"]
    # tols = range(1, 7)
    tols = [2,3,4]
    methods = ["jiggle"] #, "treescan"]
    part_sizes = [3]
    for circ in circs:
        for method in methods:
            for tol in tols:
                for m in [6]:
                    for timestep in range(10, 20):
                        # to_write = open(file_name, 'w')
                        # to_write.write(header.format(file=file, circ=circ, method=method, tol=tol, m=m, timestep=timestep))
                        # to_write.close()
                        # time.sleep(2*sleep_time)
                        dir = f"ensemble_approx_circuits_qfactor/{method}_post_gpu/{circ}"
                        circ_file = f"params_0_{6}_{tol}.pickle"
                        full_dir = join(dir, f"*", f"{timestep}")
                        full_file = join(full_dir, circ_file)
                        # print(full_file)
                        res = glob.glob(full_file)
                        if len(res) > 0:
                            print("Already Done")
                            continue

                        print(circ, method, tol)
                        args = f"python {file}.py {circ} {timestep} {method} {tol} {m} 7 6"
                        # print(args)
                        # exit(0)
                        os.system(args)
                        # output = subprocess.check_output(['python' , f"{file}.py", str(circ), str(timestep), method, str(tol), str(m), 7, 6])
                        # print(output)
                        time.sleep(sleep_time)
