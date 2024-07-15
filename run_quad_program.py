import numpy as np
import scipy.io
from qpsolvers import solve_qp, solve_ls
import matplotlib.pyplot as plt
from util import load_circuit, load_sent_unitaries
from sys import argv
import time



if __name__ == '__main__':
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])


    rep = 2
    Ms = [10000]

    circ = load_circuit(circ_name, timestep=timestep)
    Us = load_sent_unitaries(circ_name, tol)
    N = len(Us)
    # Us = np.array([U.numpy for U in Us_raw])

    V = circ.get_unitary().numpy
    d = V.shape[1]
    # Us = [U.numpy for U in Us_raw]
    
    print(f'Calculating error statistics for {N} unitaries')

    epsi = np.array([np.trace((Us[jj] - V).conj().T @ (Us[jj] - V)) for jj in range(N)])
    mean_epsi = np.mean(epsi)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(epsi, 200)
    ax.set_xlabel('eps')
    ax.set_ylabel('count')
    ax.set_title(f'eps dist {tol}', fontsize=20)

    fig.savefig(f'{circ_name}_eps_dist.png')

    print('Calculating population mean')

    EU = np.mean(Us, axis=0)

    print('Computing optimized subensemble error statistics')

    optim_val = np.zeros(len(Ms))
    optim_var = np.zeros(len(Ms))
    avg_chi1 = np.zeros(len(Ms))
    avg_chi2 = np.zeros(len(Ms))

    for mm, M in enumerate(Ms):
        error = np.zeros(rep)
        chi1 = np.zeros(rep)
        chi2 = np.zeros(rep)

        avg_qp_time = 0
        avg_ls_time_lib = 0
        avg_ls_time_np = 0
        avg_get_r_time = 0
        
        for rr in range(rep):
            tr_V_Us = np.zeros(M)
            tr_Us = np.zeros((M, M))
            sample = np.random.choice(N, M, replace=False)

            for jj in range(M):
                tr_V_Us[jj] = np.trace(V.conj().T @ Us[sample[jj]])
                for kk in range(M):
                    tr_Us[jj, kk] = np.trace(Us[sample[jj]].conj().T @ Us[sample[kk]])

            f = -2 * np.real(tr_V_Us)
            H = 2 * np.real(tr_Us)


            start = time.time()
            # ev = np.linalg.eigvals(H)
            ev = np.linalg.eigvalsh(H)
            isposdef = np.all(ev > 0)
            trials = 0
            while not isposdef and trials < 20:
                if np.abs(np.min(ev)) < 1e-10:
                    print(f'H not positive definite. Perturbing... {trials}')
                    H += 1e-10 * np.eye(M)
                    trials += 1
                else:
                    print('H not positive definite by a lot!')
                    break
                ev = np.linalg.eigvals(H)
                isposdef = np.all(ev > 0)

            if not isposdef:
                print('H not positive definite by a lot!')
                break

            Aeq = np.ones((1, M))
            beq = np.array([1])
            lbound = np.zeros(M)
            ubound = np.ones(M)

            x = solve_qp(H, f, None, None, Aeq, beq, lbound, ubound, solver='clarabel')
            avg_qp_time += (time.time() - start) / rep

            H = 2 * np.real(tr_Us)
            # Now try with LS
            start = time.time()
            trials = 0
            isposdef = False
            while not isposdef and trials < 20:
                try:
                    R = np.linalg.cholesky(H)
                    isposdef = True
                except np.linalg.LinAlgError:
                    H += 1e-10 * np.eye(M)
                    isposdef = False
                    trials += 1

            if not isposdef:
                print('H not positive definite by a lot!')
                break
            

            avg_get_r_time += (time.time() - start) / rep
            start = time.time()

            s = -1 * np.linalg.inv(R) @ f

            x_2 = solve_ls(R.T, s, None, None, Aeq, beq, lbound, ubound, solver='clarabel')

            avg_ls_time_lib += (time.time() - start) / rep + avg_get_r_time

            assert np.allclose(R @ R.T, H)

            if not np.allclose(x, x_2, rtol=1e-2, atol=1e-3):
                print('Solutions do not match!')
                fval = f @ x + 0.5 * x @ H @ x
                fval_2 = f @ x_2 + 0.5 * x_2 @ H @ x_2
                print(fval)
                print(fval_2)
                print(x)
                print(x_2)


            assert np.allclose(x, x_2, rtol=1e-2, atol=1e-3)
            assert np.allclose(np.array(np.sum(x)), [1])

            fval = f @ x + 0.5 * x @ H @ x

            # print("FVal = ", fval)
            # print("p= ", x)

            div1 = np.zeros(len(sample))
            div2 = np.zeros(len(sample) * (len(sample) - 1) // 2)
            cnt = 0
            for jj in range(len(sample)):
                div1[jj] =  np.trace(EU.conj().T @ Us[sample[jj]]) ** 2
                for kk in range(jj + 1, len(sample)):
                    div2[cnt] = np.linalg.norm(Us[sample[jj]] - Us[sample[kk]], 'fro') ** 2
                    div2[cnt] = np.trace(Us[sample[kk]].conj().T @ Us[sample[jj]]) ** 2
                    cnt += 1

            error[rr] = d
            chi1[rr] = np.mean(div1) / mean_epsi
            chi2[rr] = np.mean(div2) / mean_epsi


        print("M: ", M, " Avg QP Time: ", avg_qp_time, " Avg LS Time: ", avg_ls_time_lib, "Avg Chol Decomp Time: ", avg_get_r_time)
        optim_val[mm] = np.mean(error)
        optim_var[mm] = np.var(error)
        avg_chi1[mm] = np.mean(chi1)
        avg_chi2[mm] = np.mean(chi2)