import bp_solver
import bp_nn

def bp_sol(model, betas, error = 1e-6):
    fe_bp = []
    ener_bp = []
    m_bp = []
    m_i_bp = []
    BP_tree = bp_solver.BP_solver(model) 

    for beta in betas:

        BP_tree.converge(beta, error = error)
        fe_bp.append(BP_tree.F)
        ener_bp.append(BP_tree.E_mean)
        m_bp.append(BP_tree.M_mean)
        m_i_bp.append(BP_tree.M_i)
    
    return {
        "betas":betas,
        "fe":"fe_bp",
        "E":ener_bp,
        "M":m_bp,
        "M_i":m_i_bp
    }

def nn_sol(model, betas, bias = True, z2 = False, x_hat_clip = False,
          stats = 100, lr = 0.01,
          max_step=1000,
          batch_size=1000,
          opt = "adam"):
    fe_nn = []
    ener_nn = []
    m_nn = []
    m_i_nn = []
    net = bp_nn.bp_nn(model.N, model, bias, z2=z2, 
                    x_hat_clip=x_hat_clip )
    for beta in betas:

        net.train(beta = beta, lr=lr, max_step=max_step, 
                  batch_size=batch_size,
                 opt = opt)
        F = +1e100
        E = 0
        M = 0
        M_i = 0
        S = 0
        F_std = 0
        for ss in range(stats):
            net.compute_stat(beta, batch_size = 10000, print_=False)
            if net.F < F:
                #print(F, net.F)
                F = net.F
                E = net.E
                M = net.M
                M_i = net.M_i
                S = net.S
                F_std = net.F_std
        #net.compute_stat(beta, batch_size = 10000, print_=True)
        fe_nn.append(F)
        ener_nn.append(E)
        m_nn.append(M)
        m_i_nn.append(M_i)
        print("\nfree_energy: {0:.3f},  std_fe: {1:.5f}, mag_mean: {2:.3f}, entropy: {3:.3f} energy: {4:.3f}".format(F,
                                                    F_std,
                                                    M,
                                                    S,
                                                    E,
                                                        ), end="")

