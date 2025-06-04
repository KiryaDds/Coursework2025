from fenics import *
import numpy as np

# Set FEniCS parameters
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True
set_log_level(LogLevel.ERROR)

# Material and simulation parameters (Gold)
rho = 19.3E-6      # mass density in micrograms/micrometer^3
c = 129.0E6        # heat capacity in fJ/(ug*K)
kappa = 318.0E-3   # thermal conductivity in fJ/(um*K*ps)
tau_T = 89.286     # thermal relaxation time in ps
tau_q = 0.7438     # flux relaxation time in ps
h = 18.0E-9        # natural convection coefficient in fJ/(ps*um^2*K)
emis = 0.47        # emissivity of non-polished gold
sigma = 5.670E-17  # Stefan-Boltzmann constant in fJ/(ps*um^2*K^4)
Ta = 300.0         # ambient temperature in K

# Geometry
length = 100.0     # um
thickness = 5.0    # um
P = 30.0E3         # laser power in fJ/(um*ps)

# Mesh and function spaces
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(length, thickness, thickness), 200, 10, 10)
Space = FunctionSpace(mesh, "P", 1)
VectorSpace = VectorFunctionSpace(mesh, "P", 1)

# Markers and integration measures
cells = MeshFunction("size_t", mesh, mesh.topology().dim())
facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
da = Measure("ds", domain=mesh, subdomain_data=facets)
dv = Measure("dx", domain=mesh, subdomain_data=cells)

# Time parameters
t = 0.0
t_end = 100000.0  # ps = 0.1 microseconds
Dt = 1000.0       # time step in ps

# Initial condition
initial_T = Expression("Tini", Tini=Ta, degree=1)
T0 = interpolate(initial_T, Space)

# Laser heat source
Laser = Expression("P * exp(-1.0 * (pow(x[0] - 2, 2) + pow(x[1], 2) + pow(x[2] - 2.5, 2)))", P=P, degree=2)

# Trial and test functions
T = Function(Space)
delT = TestFunction(Space)
dT = TrialFunction(Space)
q0 = Function(VectorSpace)

# Gradients
G = as_tensor([T.dx(i) for i in range(3)])
G0 = as_tensor([T0.dx(i) for i in range(3)])

# Heat flux model (non-Fourier)
q = as_tensor([
    Dt / (Dt + tau_q) * (tau_q / Dt * q0[i] - kappa * (1 + tau_T / Dt) * G[i] + kappa * tau_T / Dt * G0[i])
    for i in range(3)
])

# Weak form
Form = (rho * c / Dt * (T - T0) * delT
        - sum(q[i] * delT.dx(i) for i in range(3))
        - rho * Laser * delT) * dvІЫ
Form += (h * (T - Ta) + emis * sigma * (T**4 - Ta**4)) * delT * da

# Jacobian
Gain = derivative(Form, T, dT)

# Output file
file_T = File("calcul/CR11/T.pvd")

# Time-stepping loop
for t in np.arange(0, t_end, Dt):
    print("Time", t)
    if t >= 2000.0:
        Laser.P = 0  # turn off laser

    solve(Form == 0, T, [], J=Gain,
          solver_parameters={"newton_solver": {"linear_solver": "mumps", "relative_tolerance": 1e-3}},
          form_compiler_parameters={"cpp_optimize": True, "representation": "quadrature", "quadrature_degree": 2})

    if t == int(t):
        file_T << (T, t)  # save output
    q0.assign(project(q, VectorSpace))
    T0.assign(T)
