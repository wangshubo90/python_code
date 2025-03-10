from math import factorial
import sympy
from functools import wraps
from time import time


def findFastInverseCoefficients(formula, x0 = 1.0, N = 20, **parameters):
    """Find the first N coefficients of the Taylor series expansion of a function about x0.
    
    Parameters:
    formula (function): The function to find the Taylor series expansion of.
    x0 (float): The point to expand the function about.
    N (int): The number of coefficients to find.
    **parameters: Additional parameters to pass to the function.
    """
    
    x = sympy.symbols('x')
    fx = formula(x, **parameters)
    
    f_n = {} # nth derivatives of f(x)
    f_n_x0 = {} # nth derivatives, but evaluated at x = x_0
    f_n[1] =  sympy.diff(fx,x,1) # differentiate f(x)
    f_n_x0[1] = f_n[1].subs({x:x0}).evalf() # do the substitution
    
    for i in range(2,N+1):
        f_n[i] = sympy.diff(f_n[i-1],x,1) 
        # it's more efficient to differentiate the previous derivative
        # once than to directly ask for the nth derivative
        f_n_x0[i] = f_n[i].subs({x:x0}).evalf()
        
    def helperP(N,Y): 
        # Y should contain N symbolic derivatives, where
        # Y[1] is the first derivative of f(x),
        # Y[2] is the second derivative of f(x), etc
        # 
        # Returns a hash table P: (i,j) -> symbolic equation
        # where P[(i,j)] = P(i,j) from the paper
        P = {}
        for j in range(1,N+1): 
            P[(j,j)] = Y[1]**j
            for k in range(j+1,N+1):
                P[(j,k)] = 0
                for l in reversed(range(1,k-j+1)): 
                    P[(j,k)] = P[(j,k)] + (l*j - k + j + l) * Y[l+1] / factorial(l+1) * P[(j,k-l)]
                P[(j,k)] = P[(j,k)]*1/(k-j) * 1/Y[1]
        return P
    
    P = helperP(N,f_n_x0) # Compute P[(i,j)] using substituted versions of f_n(x)

    b_n = {} # Vector of pre-computed dummy variable values
    b_n[1] = 1/f_n_x0[1]

    c_n = {} # vector of Taylor series coefficients
    c_n[1] = b_n[1] / factorial(1)

    for n in range(2,N+1):

        b_n[n] = 0
        for j in range(1,n): 
            b_n[n] = b_n[n] + b_n[j]/factorial(j) * P[(j,n)]
        b_n[n] = b_n[n] * factorial(n) * -1*b_n[1]**n 
        c_n[n] = b_n[n] / factorial(n)
        
    return [c_n[i] for i in range(1,N+1)]


def TaylorApprox(x, x0, y0, coefficients): 
	# x0: point about which we do expansion
	# coefficients: coefficient for each value n = 1, 2, ... 
	# y0: y(x0)
	# x: set of points at which to evaluate the Taylor Series approximation 
	y = np.zeros_like(x, dtype=np.float64)
	for i,ci in enumerate(coefficients): 
		y = (x - x0)**(i+1) * float(ci) + y
	y = y0 + y
	return y.astype(np.float64)

def fsuDisp2ForceSimpy(x, CFL = 0.0, a = 1.0, b = 1.0, urib = 1.0, rcfl = 1.0, upoly = 1.0, c1 = 1.0, c2 = 1.0):
    y = a * (sympy.exp(b * urib * upoly * x) - 1) + rcfl * urib * upoly * c2 * (1 - sympy.exp(c1 * CFL)) * x
    return y

def fsuDisp2ForceNumpy(x, CFL = 0.0, a = 1.0, b = 1.0, urib = 1.0, rcfl = 1.0, upoly = 1.0, c1 = 1.0, c2 = 1.0):
    y = a * (np.exp(b * urib * upoly * x) - 1) + rcfl * urib * upoly * c2 * (1 - np.exp(c1 * CFL)) * x
    return y


if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # CFL, a, b, urib, rcfl, upoly, c1, c2 = (0.49, 11.2, 0.1, 1.0, 1.93, 1.0, -0.68, 2.75)
    # pars = {
    #     "CFL": CFL,
    #     "a": a,
    #     "b": b,
    #     "urib": urib,
    #     "rcfl": rcfl,
    #     "upoly": upoly,
    #     "c1": c1,
    #     "c2": c2
    # }
    
    # x0 = 0.0
    # coeffs = findFastInverseCoefficients(fsuDisp2ForceSimpy, x0 = x0, N = 20,  **pars)
    # y0 = float(fsuDisp2ForceNumpy(x0, **pars))
    
    # y = np.linspace(0, 40, 100)
    # x_hat = TaylorApprox(y, y0, x0, coeffs)
    # y_ = fsuDisp2ForceNumpy(x_hat, **pars)
    # print(f"MSE: {np.mean((y_ - y)**2)}")
    # plt.figure()
    # plt.plot(y, x_hat, '-.', linewidth = 2, label = 'Taylor Approximation $\\widehat{f}^{-1}(F)$')
    # plt.plot(y_, x_hat, 'r--', linewidth = 2, label = 'Actual $f^{-1}(y)$')
    # plt.xlabel("$F (Nm)$", fontsize = 16)
    # plt.ylabel("$x = f^{-1}(F)$ (degree)", fontsize = 16)
    # plt.title("Taylor Series for Inverse Function",fontsize = 20)
    # plt.legend(fontsize = 12)
    # plt.grid(which = 'both')
    # plt.show()
    
    import pandas as pd
    df = pd.read_csv("FSUStiffness.csv")
    failed = []
    for idx, row in df.iterrows():
        sup, lvl, inf, jnt, spine, axis, cat, a, b, urib, rcfl, upoly, c1, c2 = row
        # if cat == "Normal":
        if True:
            CFL = 0.75
            if spine == "Thoracic":
                y = np.linspace(0, 10, 201)
            else:
                y = np.linspace(0, 10, 201)
            
            pars = {
                "CFL": CFL,
                "a": a,
                "b": b,
                "urib": urib,
                "rcfl": rcfl,
                "upoly": upoly,
                "c1": c1,
                "c2": c2
            }
            
            x0 = 0.0
            coeffs = findFastInverseCoefficients(fsuDisp2ForceSimpy, x0 = x0, N = 15,  **pars)
            y0 = float(fsuDisp2ForceNumpy(x0, **pars))

            x_hat = TaylorApprox(y, y0, x0, coeffs)
            y_ = fsuDisp2ForceNumpy( x_hat, **pars)
            print(f"{b * urib * upoly: 3.2f}  MSE: {np.mean((y_ - y)**2)}") 
            if np.mean((y_ - y)**2) > 1e-2:
                failed.append(row)
                # plt.figure()
                # plt.plot(y, x_hat, '-.', linewidth = 2, label = 'Taylor Approximation $\\widehat{f}^{-1}(F)$')
                # plt.plot(y_, x_hat, 'r--', linewidth = 2, label = 'Actual $f^{-1}(y)$')
                # plt.xlabel("$F (Nm)$", fontsize = 16)
                # plt.ylabel("$x = f^{-1}(F)$ (degree)", fontsize = 16)
                # plt.title("Taylor Series for Inverse Function",fontsize = 20)
                # plt.legend(fontsize = 12)
                # plt.grid(which = 'both')
                # plt.show()
                
    print(pd.DataFrame(failed).groupby(["Region", "Axis", "Category"]).count())