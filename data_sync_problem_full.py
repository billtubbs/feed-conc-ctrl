import do_mpc
import casadi as cas
import numpy as np
import pandas as pd
from platform import python_version

# Define system: single integrator
model = do_mpc.model.Model("continuous")

# Variables
x = model.set_variable(var_type="_x", var_name="x", shape=(1, 1))
u = model.set_variable(var_type="_u", var_name="u", shape=(1, 1))

# Dynamics: dx/dt = u
model.set_rhs("x", u)

# Output: y = x (should equal state)
model.set_meas(meas_name="y", expr=x)

model.setup()

# Sampling period
t_step = 1.0

# Define state estimator
estimator = do_mpc.estimator.EKF(model=model)
# estimator = do_mpc.estimator.StateFeedback(model)
estimator.settings.t_step = t_step
estimator.setup()
Q = cas.DM([[0.01 ** 2]])  # process noise
R = cas.DM([[0.1 ** 2]])  # measurement noise

# Define controller
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    "n_horizon": 5,
    "t_step": t_step,
    "store_full_solution": True,
}
mpc.set_param(**setup_mpc)
# Set point tracking objective
mterm = (model.x["x"] - 1.0) ** 2  # terminal cost
lterm = (model.x["x"] - 1.0) ** 2  # stage cost
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(u=0.1)  # control input penalty
mpc.setup()

# Define simulator
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=t_step)
simulator.setup()

# Initial condition
x0 = np.array([[0.0]])
simulator.x0 = x0
estimator.x0 = x0
mpc.x0 = x0

# Prime the objects
simulator.set_initial_guess()
estimator.set_initial_guess()
mpc.set_initial_guess()

print("Running simulation...")
for k in range(5):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next=y_next, u_next=u0, Q_k=Q, R_k=R)

# How I usually run simulations:
# simulator.xk = x0
# estimator.xk = x0
#
# X = []
# Y = []
# X_est = []
# U = []
# for k in range(n_steps + 1):
#     yk = simulator.yk  # measurements
#     xk = estimator.update(yk)  # update step
#     uk = mpc(xk)
#
#     # Save results
#     X.append(simulator.xk)
#     Y.append(yk)
#     X_est.append(xk)
#     U.append(uk)
#
#     simulator.step(uk)  # simulation step
#     estimator.step(uk)  # prediction step

# Display Results
data_fields = ["_time", "_u", "_x", "_y"]
data_objects = {"mpc": mpc, "simulator": simulator, "estimator": estimator}
sim_results = pd.concat(
    {
        name: pd.concat(
            {
                var_name: pd.DataFrame(item.data[var_name])
                for var_name in data_fields
            },
            axis=1,
        )
        for name, item in data_objects.items()
    },
    axis=1,
)

print("Simulation Results:")
print(sim_results.round(2))

# Display version info
print(f"\nPython: {python_version()}")
print(f"CasADi: {cas.__version__}")
print(f"Do-MPC: {do_mpc.__version__}")

breakpoint()
