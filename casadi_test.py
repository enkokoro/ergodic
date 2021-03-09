import casadi
import numpy as np 
import itertools

C = [np.array([1, 1, 1]),
     np.array([1, 7, 1, 7, 4]), 
     np.array([2, -4, 7]), 
     np.array([1.2, 1.1, 0.9]),
     np.array([3, -10, 2, 5, -4, 30, 0, 1])]

def sqr_metric(x):                                              # doesn't seem to make sparsity have large effect
    return x**2
def oneminussqrt_metric(x):
    return 1 - (x-1)**2
def other_metric(x):
    return 1 - 1/(100*(x+0.01))

metrics = [sqr_metric, oneminussqrt_metric, other_metric]

for (c, elt_metric) in itertools.product(C, metrics):
    print("="*120)
    print("Metric ", elt_metric.__name__)
    o = np.ones(len(c))
    om = np.ones((len(c), len(c)))
    eps = 1e-3
    k = 3 # number nonzero
    print("Averaging ", c)
    opti = casadi.Opti()
    x = opti.variable(len(c), len(c), 'symmetric')

    opti.subject_to( casadi.vec(x) >= 0 )                       # elements nonnegative
    opti.subject_to( casadi.vec(x) <= 1 )                       # elements bounded by 1
    opti.subject_to( x@o == 1 )                                 # rows sum to 1

    # opti.subject_to( casadi.eig_symbolic(A) <= np.sqrt(k) )   # minimize eigenvalue - eigenvalue vs sparsity 
                                                                # https://math.stackexchange.com/questions/199268/eigenvalues-vs-matrix-sparsity
    # opti.subject_to( casadi.sum1(casadi.sum2(x)) <= 0.3*9 )
    # opti.subject_to( (x.nnz() / x.numel()) <= 0.5 )           # calculating sparsity directly
                                                                # some reason it defaults to dense matrix

    opti.set_initial(x, np.diag(np.ones(len(c))))

    rowsum = x@o

    csum = casadi.sum1(c)*o
    avmetric = casadi.sum1(((x@c - csum)**2))                   # |Xc - average(c)| - this shows for one c, but there will be more c_k

    # sparsemetric = x.nnz() / x.numel()                        # some reason it defaults to dense matrix
    
    sparsemetric = casadi.sum1(casadi.sum2(elt_metric(x)))/x.numel()    # sparsity metric - average penalization over elements

    opti.minimize(avmetric + sparsemetric)

    p_opts = {}
    s_opts = {'print_level': 0}
    opti.solver('ipopt', p_opts, s_opts)
    sol = opti.solve()
    
    print("Solution \n", sol.value(x))

    r = 2
    rounded = sol.value(x).round(r)
    print("Rounded ", r, " \n", rounded)
    print("Sparsity ", np.count_nonzero(rounded)/rounded.size)

    print("Calculated 'Average' from rounded ", rounded@c)
    print("Actual Average ", np.average(c))