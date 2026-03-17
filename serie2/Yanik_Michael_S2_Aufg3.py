import sympy as sp
import numpy as np
sp.init_printing()

# var
x1, x2, x3 = sp.symbols('x1 x2 x3')

f = sp.Matrix([
    x1 + x2**2 - x2**3 - 13,                # f1
    sp.ln(x2**4) + sp.exp(0.5*x3 - 1) - 1,  # f2  
    1/(x2 - 3)**2 - x3**3 + 7               # f3
])

X = sp.Matrix([x1, x2, x3])

# Linearisierungspunkt x0 = (1.5, 3, 2.5)
x0 = sp.Matrix([sp.Rational(3,2), 3, sp.Rational(5,2)])  # 1.5=3/2, 2.5=5/2
print(x0.T)

# 1. Funktion im Punkt auswerten: f(x0)
f_x0 = f.subs([(x1,x0[0]), (x2,x0[1]), (x3,x0[2])])
print(f_x0)
print("Numerisch:")
print(f_x0.evalf())

# 2. Jacobi-Matrix Df
Df = f.jacobian(X)
print("Jacobi-Matrix Df(x):")
print(Df)

# 3. Df auswerten: Df(x0)
Df_x0 = Df.subs([(x1,x0[0]), (x2,x0[1]), (x3,x0[2])])
print("Df(x0):")
print(Df_x0)
print("Numerisch:")
print(Df_x0.evalf())

# Symbolische Linearisierung (mit Symbol h = x - x0)
h1, h2, h3 = sp.symbols('h1 h2 h3')
h = sp.Matrix([h1, h2, h3])

L = f_x0 + Df_x0 * h
print(L)
print("Numerisch:")
print(L.evalf())
