from util import get_completed_blocks, get_circ_data

cliff_t = True
if __name__ == '__main__':
    circ_names = get_completed_blocks(cliff_t)
    # circ_names = ["qae11"]
    err_thresholds = [0.5, 1e-6]
    for circ_name in circ_names:
        for err_threshold in err_thresholds:
            count = get_circ_data(circ_name, err_threshold, cliff_t=cliff_t)
            print(circ_name, err_threshold, count)