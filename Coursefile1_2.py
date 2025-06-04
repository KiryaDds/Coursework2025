from fenics import *
import numpy as np

# General FEniCS parameters
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Material and simulation parameters
rho = 7860.0          # Density (kg/m^3)
c = 624.0             # Specific heat capacity (J/(kg·K))
kappa = 30.1          # Thermal conductivity (W/(m·K))
h = 0.0               # Heat convection coefficient (W/(m^2·K))
Ta = 300.0            # Ambient temperature (K)
L = 0.1               # Length in x and y directions (m)
thickness = 0.001     # Plate thickness (m)
P = 3.0e6             # Laser power (W/kg)
speed = 0.02          # Laser movement speed (m/s)

# Time parameters
t_end = 50.0
dt = 0.1

# Mesh and function space
mesh = BoxMesh(Point(0, 0, 0), Point(L, L, thickness), 200, 200, 2)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary measure
ds = Measure('ds', domain=mesh)

# Initial temperature distribution
T0 = interpolate(Constant(Ta), V)

# Define trial and test functions
T = TrialFunction(V)
v = TestFunction(V)

# Left (linear) part of the variational form
a = (rho * c / dt) * T * v * dx + kappa * dot(grad(T), grad(v)) * dx + h * T * v * ds
A = assemble(a)


# Time-dependent laser heat source
Laser = Expression(
    "P * exp(-50000*(pow(x[0] - 0.5*l*(1 + 0.5*sin(2*pi*t/te)), 2) + pow(x[1] - velo*t, 2)))",
    P=P, t=0, te=t_end / 10.0, l=L, velo=speed, degree=2
)

T_sol = Function(V)  # Solution function
file_T = File("~/T.pvd")  # Output file for temperature

v_ = TestFunction(V)  # Test function for RHS assembly

# Time-stepping loop
for t in np.arange(0, t_end + dt, dt):
    print("Time:", t)
    Laser.t = t  # Update laser time parameter

    # Right-hand side depending on time, includes previous solution and heat source
    L_form = (rho * c / dt) * T0 * v_ * dx + rho * Laser * v_ * dx + h * Ta * v_ * ds

    b = assemble(L_form)  # Assemble RHS vector for current time step
    solve(A, T_sol.vector(), b, "cg")  # Solve linear system

    # Save solution at integer time steps
    if abs(t - round(t)) < 1e-8:
        file_T << (T_sol, t)

    T0.assign(T_sol)  # Update previous solution for next time step
