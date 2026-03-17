import sympy as sp
import numpy as np
sp.init_printing()

# Deklarierte Variablen
x, y, z = sp.symbols('x y z')

# Erster Ausdruck
f1 = x**2 + y - 3*z
print(f1)

# exp, log, sin, cos, tan vorhanden
f2 = sp.exp(y) -sp.sin(z)
print(f2)

# Substitution
print(f1.subs(x, 2))
print(f2.subs(y, f1))

# Substitution mehrerer Variabeln
h = f2.subs([(y,1),(z,5)])
print(h)

# Numerische Auswertung
print(h.evalf())

# Ableitung
print(sp.diff(f2,z))

# sp.Matrix für vektorwertige Funktionen f(x,y,z) = (f1,f2)
f = sp.Matrix([f1,f2])
print(f)

# Bestimmung der Jacobimatrix Df = df/dX, wobei X = (x,y,z) [Vektor der Variablen]
X = sp.Matrix([x,y,z])
Df = f.jacobian(X)
print(Df)

# Substitution von Variablen in der Jacobi Matrix durch Werte
Df0 = Df.subs([(x,3),(y,4),(z,5)])
print(Df0)
# Numerische Auswertung der Jacobimatrix
print(Df0.evalf())

# lambdify: Umwandlung von symbolischen Ausdrücken in numerische Funktionen
func1 = sp.lambdify((x,y,z), f)
print(func1(7,8,9))

# Einfachere Variante: lambdify direkt mit Matrizen
# Umwandlung in numerische Funktionen, die dann mit numpy Arrays arbeiten können.
func = sp.lambdify([(x,y,z)], f, "numpy")
jac = sp.lambdify([(x,y,z)], Df, "numpy")
v = np.array([7,8,9])
print(func(v))
print(jac(v))