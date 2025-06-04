"""
Computational Reality 10: Temperature distribution in a macroscopic body
Author: B. Emek Abali
License: GNU GPL Version 3.0 or later
http://www.gnu.org/licenses/gpl-3.0.en.html
"""

from fenics import *
import numpy as np

# General FEniCS parameters
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True
#set_log_level(ERROR)

# Material and simulation parameters
rho = 7860.0         # Mass density of steel (kg/m^3)
c = 624.0            # Specific heat capacity (J/(kg·K))
kappa = 30.1         # Thermal conductivity (W/(m·K))
#h = 18.0             # Heat convection coefficient (W/(m^2·K))
h = 0.0
Ta = 300.0           # Ambient temperature (K)
L = 0.1              # Length in x and y directions (m)
thickness = 0.001    # Plate thickness (m)
P = 3.0e6            # Laser power (W/kg)
speed = 0.02         # Laser movement speed (m/s)

# Time parameters
t = 0.0
t_end = 50.0
dt = 0.1

# Mesh and function space
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(L, L, thickness), 200, 200, 2)
V = FunctionSpace(mesh, 'P', 1)

# Subdomain markers (not actively used here, but ready for extension)
cells = MeshFunction("size_t", mesh, mesh.topology().dim())
facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
da = Measure("ds", domain=mesh, subdomain_data=facets)
dv = Measure("dx", domain=mesh, subdomain_data=cells)

# Initial condition
initial_T = Expression("Tini", Tini=Ta, degree=1)
T0 = interpolate(initial_T, V)

# Laser source term (time-dependent)
Laser = Expression(
    "P * exp(-50000.0 * (pow(x[0] - 0.5 * l * (1 + 0.5 * sin(2 * pi * t / te)), 2) + pow(x[1] - velo * t, 2)))",
    P=P, t=0, te=t_end / 10.0, l=L, velo=speed, degree=2
)

# Variational problem
T = TrialFunction(V)
v = TestFunction(V)

F = (
    (rho * c / dt) * (T - T0) * v * dv
    + kappa * dot(grad(T), grad(v)) * dv
    - rho * Laser * v * dv
    + h * (T - Ta) * v * da
)

a = lhs(F)
L_form = rhs(F)

# Preassembly of time-independent matrix
A = assemble(a)
b = None

# Solution function and output file
T_sol = Function(V)
file_T = File("~/T.pvd")

# Time-stepping loop
for t in np.arange(0, t_end, dt):
    print("Time:", t)
    Laser.t = t
    b = assemble(L_form, tensor=b)
    solve(A, T_sol.vector(), b, "cg")
    if abs(t - round(t)) < 1e-8:  # Save once per integer time step
        file_T << (T_sol, t)
    T0.assign(T_sol)
