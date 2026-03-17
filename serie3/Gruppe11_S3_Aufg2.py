import  sympy as sp
import numpy as np

x, y = sp.symbols('x y')
f1 = x**2 / 186**2 - y**2 / (300**2 - 186**2) -1
f2 = (y-500)**2 / 279 **2 - (x-300)**2/(500**2-279**2) -1
p1 = sp.plot_implicit(sp.Eq(f1,0), (x,-2000,2000),(y,-2000,2000), show=False)
p2 = sp.plot_implicit(sp.Eq(f2,0), (x,-2000,2000),(y, -2000,2000), show=False)
p1.append(p2[0])
p1.show()

# aufgabe B)

f_a = sp.Matrix([f1, f2])
Df = f_a.jacobian([x, y])

func = sp.lambdify([[[x], [y]]], f_a, "numpy")
jac = sp.lambdify([[[x], [y]]], Df, "numpy")

def newton(x0):
    x1 = np.linalg.solve(jac(x0), -func(x0)) + x0
    while np.linalg.norm(func(x1), 2) > 10e-5:
        x0 = x1
        x1 = np.linalg.solve(jac(x0), -func(x0)) + x0
    return x1

# Schnittpunkte
x0 = np.array([[-200], [87]])
print(newton(x0))
x0 = np.array([[250], [225]])
print(newton(x0))
x0 = np.array([[725], [925]])
print(newton(x0))
x0 = np.array([[-1273], [1600]])
print(newton(x0))
