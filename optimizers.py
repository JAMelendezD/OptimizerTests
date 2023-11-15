import numpy as np
from numpy import exp, sqrt, cos, pi, e, sin

def Himmelblau(x, y):
    return (x**2+y-11)**2 + (x+y**2-7)**2

def Rosenbrock(x,y):
    return (1-x)**2 + 100*(y-x**2)**2

def Ackley(x, y):
    return -20.0*exp(-0.2*sqrt(0.5*(x**2+y**2)))-exp(0.5*(cos(2*pi*x)+cos(2*pi*y)))+e+20

def Goldstein(x, y):
    return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))  

def DropWave(x, y):
    return -(1+np.cos(12*np.sqrt(x**2+y**2))) / (0.5*(x**2+y**2)+2)

def EggHolder(x, y):
    a=sqrt(abs(y+x/2+47))
    b=sqrt(abs(x-(y+47)))
    c=-(y+47)*sin(a)-x*sin(b)
    return c

def Michalewicz(x, y):
    return -1*((sin(x)*np.sin((1*x**2)/np.pi)**20)+(np.sin(y)*np.sin((2*y**2)/np.pi)**20))

def Booth(x, y):
    return (x+2*y-7)**2 + (2*x+y-5)**2

def Levy(x, y):
    return sin(3*pi*x)**2 + (x-1)**2*(1+sin(3*pi*y)*sin(3*pi*y))+ (y-1)*(y-1)*(1+sin(2*pi*y)*sin(2*pi*y))

def update_simplex(f, eps, maxiter, xmin, xmax, out, alpha = 1, gamma = 2, sigma = 0.5, rho = 0.5):
    
    n = 3
    simplex = np.random.uniform(xmin, xmax, (n, 2))
    evals = np.zeros(n)
    
    for i in range(n):
        vertex = simplex[i]
        evals[i] = f(vertex[0], vertex[1])

    iters = 0
    step = "START"
    traj = []

    while True:

        order_ind = evals.argsort()
        simplex = simplex[order_ind]
        evals = evals[order_ind]

        center = np.sum(simplex[:-1,:], axis = 0) / (n-1)
        rmsd = np.linalg.norm(center-simplex)
        out.write(f"{step}\n")
        traj.append(simplex)
        
        if iters >= maxiter:
            out.write(f"END MAX_ITERS: {rmsd:8.6f}{iters:10d}{evals[0]:8.4f} {str(simplex[0]):16s}\n")
            break
        
        if rmsd <= eps:
            out.write(f"END RMSD: {rmsd:8.6f}{iters:10d}{evals[0]:8.4f} {str(simplex[0]):16s}\n")
            break
        
        iters += 1
        
        ### REFLECTION ###
        
        reflected = (center + alpha * (center - simplex[n-1]))
        test_ref = f(reflected[0], reflected[1])
        
        if evals[0] <= test_ref < evals[n-2]: # not the best but better than second worst
            simplex[n-1] = reflected # keep reflected
            evals[n-1] = test_ref
            step = 'REFLECT'
            continue
            
        ### EXPANSION ###            
            
        if test_ref < evals[0]: # best point
            expanded = (center + gamma * (reflected - center))
            test_exp = f(expanded[0], expanded[1])
            if test_exp < test_ref: # expanded better than reflected keep expanded
                simplex[n-1] = expanded
                evals[n-1] = test_exp
                step = 'EXPAND'
                continue
            else:
                simplex[n-1] = reflected # keep reflected
                evals[n-1] = test_ref
                step = 'REFLECT'
                continue
        
        #### CONTRACTION ###
        
        if test_ref < evals[n-1]:
            contracted = (center + rho * (reflected - center))
            test_con = f(contracted[0], contracted[1])
            if test_con < test_ref: # contracted better than reflected
                simplex[n-1] = contracted # keep contracted
                evals[n-1] = test_con
                step = 'CONTRACT'
                continue
            else:
                for i in range(1,n): # change all except the best
                    simplex[i] = (simplex[0] + sigma * (simplex[i] - simplex[0]))
                    evals[i] = f(simplex[i][0], simplex[i][1])
                    step = 'SHRINK'
                    continue
                
        if test_ref >= evals[n-1]:
            contracted = (center + rho * (simplex[n-1] - center))
            test_con = f(contracted[0], contracted[1])
            if test_con < evals[n-1]: # contracted better worst point
                simplex[n-1] = contracted # keep contracted
                evals[n-1] = test_con
                step = 'CONTRACT'
                continue
            else: # reflected is the worst
                for i in range(1,n): # change all except the best
                    simplex[i] = (simplex[0] + sigma * (simplex[i] - simplex[0]))
                    evals[i] = f(simplex[i][0], simplex[i][1])
                    step = 'SHRINK'
                    continue
                
    return np.array(traj)

def run(errorfunc, pmin, pmax):

    for i in range(10):
        eps = 0.01
        max_iter = 20
        log = open(f"./data/simplex_{i:d}.txt", 'w')
        traj = update_simplex(errorfunc, eps, max_iter, pmin, pmax, log)
        log.close()
        np.save(f"./data/traj_{i:d}", traj, allow_pickle=True)

run(Levy, -5, 5)
