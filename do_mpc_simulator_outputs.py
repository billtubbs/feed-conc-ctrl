import do_mpc
import casadi as cas
import numpy as np
import pandas as pd
from platform import python_version

# Minimal System: Integrator
model = do_mpc.model.Model('continuous')

# Variables
x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))

# Dynamics: dx/dt = u
model.set_rhs('x', u)

# Output: y = x (should equal state)
model.set_meas(meas_name='y', expr=x)

model.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=1.0)
simulator.setup()

# Initial condition
x0 = np.array([[0.0]])
simulator.x0 = x0

# Apply constant input
u_constant = np.array([[1.0]])

print("Expected behavior with dx/dt = u and y = x:")
print("If u = 1.0 (constant), then x should increase by 1.0 each step")
print("If y[k] = x[k], then y should equal x at each time")
print("If y[k] = x[k+1], then y[k] will equal x[k+1]\n")

# Simulate 5 steps
for k in range(5):
    y_k = simulator.make_step(u_constant)

# ========================================
# Display Results
# ========================================
sim_results = pd.concat({
    't': pd.DataFrame(simulator.data['_time']), 
    'u': pd.DataFrame(simulator.data['_u']),
    'x': pd.DataFrame(simulator.data['_x']),
    'y': pd.DataFrame(simulator.data['_y'])
}, axis=1)

print("Simulation Results:")
print(sim_results.round(2))

# Display version info
print(f"\nPython: {python_version()}")
print(f"CasADi: {cas.__version__}")
print(f"Do-MPC: {do_mpc.__version__}")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)

print("\nChecking if y[k] = x[k] (output computed from current state):")
for k in range(5):
    t_k = sim_results.loc[k, ('t', 0)]
    x_k = sim_results.loc[k, ('x', 0)]
    y_k = sim_results.loc[k, ('y', 0)]
    match = abs(x_k - y_k) < 0.01
    print(f"  k={k}: t={t_k:.1f}, x[k]={x_k:.2f}, y[k]={y_k:.2f}, Match={match}")

print("\nChecking if y[k] = x[k+1] (output computed from next state):")
for k in range(4):
    t_k = sim_results.loc[k, ('t', 0)]
    x_k_plus_1 = sim_results.loc[k+1, ('x', 0)]
    y_k = sim_results.loc[k, ('y', 0)]
    match = abs(x_k_plus_1 - y_k) < 0.01
    print(f"  k={k}: t={t_k:.1f}, x[k+1]={x_k_plus_1:.2f}, y[k]={y_k:.2f}, Match={match}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
if abs(sim_results.loc[0, ('x', 0)] - sim_results.loc[0, ('y', 0)]) < 0.01:
    print("✓ y[k] = x[k]: Output computed from current state")
elif abs(sim_results.loc[1, ('x', 0)] - sim_results.loc[0, ('y', 0)]) < 0.01:
    print("✓ y[k] = x[k+1]: Output computed from NEXT state (after integration)")
    print("\nThis means at row k:")
    print("  - t = k")
    print("  - u = u[k] (input applied at time k)")
    print("  - x = x[k+1] (state AFTER applying u[k])")
    print("  - y = y[k] = h(x[k+1]) (output from NEW state)")
else:
    print("? Unclear alignment pattern")
