from util.gg import get_approx_t_str
from util.run_gg_circ import worker
import multiprocessing as mp
import numpy as np
import time
import gc
import psutil


# def process_args(ang: float, prec: int, task_queue, mailboxes):
#     # Get process id
#     process_id = mp.current_process().pid
#     # Create a task for the worker
#     task_queue.put((process_id, ang, prec))

#     # Wait for the result
#     result_queue = mailboxes[process_id % 250]
#     while result_queue.empty():
#         time.sleep(0.1)

#     # Get the result
#     result = result_queue.get()
#     return result

def double_inputs():
    while True:
        x = yield
        yield x * 2

if __name__ == '__main__':
    # manager = mp.Manager()
    # task_queue = manager.Queue()
    # mailboxes = manager.dict() 

    # # For every process, create a mailbox
    # for i in range(250):
    #     mailboxes[i] = manager.Queue()

    # # Start worker process
    # worker_process = mp.Process(target=worker, args=(task_queue, mailboxes))
    # worker_process.start()

    # test_gen = double_inputs()

    # next(test_gen)
    # print(test_gen.send(5))
    # next(test_gen)
    # print(test_gen.send(10))


    # # Create a list of angles
    num_tasks = 100
    angles = np.random.uniform(0, 2*np.pi, num_tasks)
    random_precisions = np.random.randint(5, 15, num_tasks)

    # # Manager pool
    # # pool = manager.Pool(processes=250)
    # # args = zip(angles, random_precisions, [task_queue]*num_tasks, [mailboxes]*num_tasks)
    # # start = time.time()
    # # results = pool.starmap(process_args, args)
    # # end = time.time()

    # # print(f"Time taken: {end-start}")

    # # task_queue.put("exit")
    # # worker_process.join()

    start = time.time()
    # args = list(zip(angles, random_precisions))
    # # # print(args[0])
    # with mp.Pool(250) as pool:
    #     results = pool.starmap(get_approx_t_str, args)
    process = psutil.Process()
    # print(process.memory_info().rss)  # in bytes 
    start_mem = process.memory_info().rss
    for _ in range(100):
        result = get_approx_t_str(angles[0], random_precisions[0])
        # pass
    end = time.time()

    # process = psutil.Process()
    gc.collect()
    # print(process.memory_info().rss)  # in bytes 
    final_mem = process.memory_info().rss
    print(f"Memory used: {final_mem - start_mem}")
    if final_mem - start_mem > 10000:
        print("Memory leak")
        exit(1)
    # print(f"Time taken: {end-start}")
    # print(results)
