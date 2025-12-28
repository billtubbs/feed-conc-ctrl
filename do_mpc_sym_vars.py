# Compare different ways of accessing symbolic variables

import do_mpc
import casadi as cas

# Minimal example showing symbols change after model.setup()
model = do_mpc.model.Model('continuous')

# Add variables (capture returned symbols)
x1_from_def = model.set_variable(var_type='_x', var_name='x1', shape=(1, 1))
u1_from_def = model.set_variable(var_type='_u', var_name='u1', shape=(1, 1))

print("FROM set_variable():")
print(f"  x1_from_def: {x1_from_def}")
print(f"  ID: {id(x1_from_def)}")

# Get symbols BEFORE setup
x1_before_setup = model.x['x1']
u1_before_setup = model.u['u1']

print("\nBEFORE model.setup():")
print(f"  model.x['x1']: {x1_before_setup}")
print(f"  ID: {id(x1_before_setup)}")

# Create expression with pre-setup symbol
expr_before_setup = x1_before_setup**2 + u1_before_setup

# Set RHS (required for setup)
model.set_rhs('x1', u1_before_setup)

# Call setup
model.setup()

# Get symbols AFTER setup
x1_after_setup = model.x['x1']
u1_after_setup = model.u['u1']

print("\nAFTER model.setup():")
print(f"  model.x['x1']: {x1_after_setup}")
print(f"  ID: {id(x1_after_setup)}")

print("\n" + "="*60)
print("COMPARISON OF ALL x1 SYMBOLS:")
print("="*60)

print("\n1. x1_from_def vs x1_before_setup:")
print(f"   Symbolically equal: {cas.is_equal(x1_from_def, x1_before_setup)}")
print(f"   Same object: {x1_from_def is x1_before_setup}")

print("\n2. x1_from_def vs x1_after_setup:")
print(f"   Symbolically equal: {cas.is_equal(x1_from_def, x1_after_setup)}")
print(f"   Same object: {x1_from_def is x1_after_setup}")

print("\n3. x1_before_setup vs x1_after_setup:")
print(f"   Symbolically equal: {cas.is_equal(x1_before_setup, x1_after_setup)}")
print(f"   Same object: {x1_before_setup is x1_after_setup}")

print("\n" + "="*60)

# The problem: Using pre-setup expression in MPC objective
mpc = do_mpc.controller.MPC(model)
mpc.set_param(n_horizon=5, t_step=1.0)

print("\nAttempting to use pre-setup expression in objective:")
try:
    mpc.set_objective(mterm=cas.DM(0), lterm=expr_before_setup)
    mpc.set_rterm(u1=1.0)  # Avoid warning
    mpc.setup()
    print("  ✓ Pre-setup expression works")
except Exception as e:
    print(f"  ✗ Pre-setup expression FAILED: {type(e).__name__}")
    print(f"     {str(e)[:150]}")

# The solution: Rebuild expression after setup
mpc2 = do_mpc.controller.MPC(model)
mpc2.set_param(n_horizon=5, t_step=1.0)

print("\nUsing post-setup symbols in objective:")
expr_after_setup = x1_after_setup**2 + u1_after_setup
try:
    mpc2.set_objective(mterm=cas.DM(0), lterm=expr_after_setup)
    mpc2.set_rterm(u1=1.0)  # Avoid warning
    mpc2.setup()
    print("  ✓ Post-setup expression works")
except Exception as e:
    print(f"  ✗ Post-setup expression FAILED: {type(e).__name__}")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print("BEFORE model.setup(): model.x['x1'] returns a TEMPORARY symbol")
print("                      that is NOT the same as the original from set_variable()")
print()
print("AFTER model.setup():  model.x['x1'] returns the ORIGINAL symbol")
print("                      from set_variable() (symbolically equal)")
print()
print("KEY INSIGHT: Accessing model.x BEFORE setup gives temporary symbols.")
print("             These temporary symbols will FAIL in MPC set_objective().")
print()
print("SOLUTION: Always rebuild expressions using model.x AFTER model.setup()")