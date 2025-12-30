"""Functions for constructing MPC controllers from CasADi system models."""

import warnings

import casadi as cas
import do_mpc


def construct_mpc(
    system,
    control_design,
    mpc_params,
    cv_weights,
    setpoints=None,
    mv_weights=None,
    bounds=None,
):
    """
    Construct a Do-MPC controller from a CasADi continuous-time system model.

    Parameters
    ----------
    system : StateSpaceModelCT
        Continuous-time CasADi system model with attributes:
        - state_names: list of state variable names
        - input_names: list of input variable names
        - output_names: list of output variable names
        - f: state transition function
        - h: output function

    control_design : dict
        Dictionary defining the control structure with keys:
        - 'states': (optional) list of state names to include in MPC
                   model. If not provided, all states from
                   system.state_names are used.
        - 'manipulated_variables': list of MV names
                   (subset of system.input_names)
        - 'unmeasured_disturbances': list of disturbance names
                   (subset of system.input_names)
        - 'controlled_variables': list of CV names
                   (subset of system.output_names)
        - 'measured_disturbances': (optional) list of measured
                   disturbance names

    mpc_params : dict
        MPC setup parameters with keys:
        - 't_step': time step for discretization (required)
        - 'n_horizon': prediction horizon steps (required)
        - 'n_robust': robust horizon (optional, default: 0)
        - 'store_full_solution': whether to store full solution
                   (optional, default: True)

    cv_weights : dict
        Dictionary of tracking weights for controlled variables.
        Keys are CV names, values are weights (higher = more important).
        All controlled variables must have weights specified.

    setpoints : dict, optional
        Dictionary of setpoints for controlled variables.
        Keys are CV names, values are target values.
        If None, all setpoints default to 0.

    mv_weights : dict, optional
        Dictionary of control effort weights for manipulated variables.
        Keys are MV names, values are weights for penalizing control
        effort. If None, no control effort penalty term is added to
        the cost function.

    bounds : dict, optional
        Dictionary of variable bounds with structure:
        {
            'states': {
                'var_name': {'lower': value, 'upper': value},
                ...
            },
            'inputs': {
                'var_name': {'lower': value, 'upper': value},
                ...
            },
            'disturbances': {
                'var_name': {'lower': value, 'upper': value},
                ...
            },
            'outputs': {
                'var_name': {'lower': value, 'upper': value},
                ...
            }
        }
        Output bounds are implemented using nonlinear constraints
        via set_nl_cons(). If None, no bounds are set (unbounded).

    Returns
    -------
    mpc : do_mpc.controller.MPC
        Configured and setup MPC controller ready for use.
        Call mpc.make_step(x0) to get control actions.

    model : do_mpc.model.Model
        The underlying do-mpc model used by the controller.

    Examples
    --------
    >>> control_design = {
    ...     "manipulated_variables": ['tank_1_v_dot_in', 'tank_2_v_dot_in'],
    ...     "unmeasured_disturbances": ['tank_1_conc_in', 'tank_2_conc_in'],
    ...     "controlled_variables": ['tank_1_L', 'mixer_conc_out'],
    ... }
    >>>
    >>> mpc_params = {
    ...     't_step': 1.0,
    ...     'n_horizon': 50,
    ... }
    >>>
    >>> setpoints = {
    ...     'tank_1_L': 1.5,
    ...     'mixer_conc_out': 2.0,
    ... }
    >>>
    >>> cv_weights = {
    ...     'tank_1_L': 0.1,
    ...     'mixer_conc_out': 10.0,
    ... }
    >>>
    >>> bounds = {
    ...     'inputs': {
    ...         'tank_1_v_dot_in': {'lower': 0.0, 'upper': 2.0},
    ...         'tank_2_v_dot_in': {'lower': 0.0, 'upper': 2.0},
    ...     },
    ...     'states': {
    ...         'tank_1_L': {'lower': 0.1, 'upper': 3.0},
    ...     },
    ...     'outputs': {
    ...         'mixer_conc_out': {'lower': 1.5, 'upper': 2.5},
    ...     }
    ... }
    >>>
    >>> mpc, model = construct_mpc(
    ...     system,
    ...     control_design=control_design,
    ...     mpc_params=mpc_params,
    ...     setpoints=setpoints,
    ...     cv_weights=cv_weights,
    ...     bounds=bounds
    ... )
    """

    # Get state names from control_design or use all system states
    state_names = control_design.get('states', system.state_names)

    # Validate control_design
    all_inputs = set(
        control_design.get("manipulated_variables", []) +
        control_design.get("unmeasured_disturbances", []) +
        control_design.get("measured_disturbances", [])
    )
    if all_inputs != set(system.input_names):
        raise ValueError(
            f"Control design inputs {all_inputs} do not match "
            f"system inputs {set(system.input_names)}"
        )

    # Validate state names - must be same set (can be reordered)
    if set(state_names) != set(system.state_names):
        raise ValueError(
            f"Control design states {set(state_names)} do not match "
            f"system states {set(system.state_names)}. "
            f"Currently, all states must be included (reordering is allowed)."
        )
    # TODO: Support selecting a subset of states by creating dummy symbolic
    # values for non-selected states and checking if resulting expressions
    # contain any free variables

    # Set default parameters for optional arguments
    if setpoints is None:
        setpoints = {
            cv: 0.0 for cv in control_design["controlled_variables"]
        }

    if bounds is None:
        bounds = {}

    # Validate cv_weights
    for cv_name in cv_weights.keys():
        if cv_name not in control_design["controlled_variables"]:
            raise ValueError(
                f"cv_weight '{cv_name}' is not in "
                f"control_design['controlled_variables']: "
                f"{control_design['controlled_variables']}"
            )

    # Warn if setpoints are specified without corresponding weights
    for sp_name in setpoints.keys():
        if sp_name not in cv_weights or cv_weights[sp_name] == 0:
            warnings.warn(
                f"Setpoint specified for '{sp_name}' but no "
                f"corresponding cv_weight (or weight is 0). "
                f"This setpoint will have no effect on the cost function.",
                UserWarning
            )

    # ========================================
    # 1. Create do-mpc model
    # ========================================
    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)

    # Add manipulated variables (MVs)
    for name in control_design["manipulated_variables"]:
        model.set_variable(var_type='_u', var_name=name, shape=(1, 1))

    # Add state variables (can be in any order specified by control_design)
    for name in state_names:
        model.set_variable(var_type='_x', var_name=name, shape=(1, 1))

    # Augment model with additional states for unmeasured disturbances
    for name in control_design.get("unmeasured_disturbances", []):
        model.set_variable(var_type='_x', var_name=name, shape=(1, 1))

    # Augment model with additional states for measured disturbances
    for name in control_design.get("measured_disturbances", []):
        model.set_variable(var_type='_x', var_name=name, shape=(1, 1))

    # ========================================
    # 2. Build state and input vectors
    # ========================================
    t = 0  # assume time invariant

    # Build state vector in system.state_names order
    # (required by system.f and system.h)
    states = cas.vcat([model.x[name] for name in system.state_names])

    inputs = []
    for name in system.input_names:
        if name in control_design['manipulated_variables']:
            inputs.append(model.u[name])
        elif name in control_design.get('measured_disturbances', []):
            inputs.append(model.x[name])
        elif name in control_design.get('unmeasured_disturbances', []):
            inputs.append(model.x[name])
    inputs = cas.vcat(inputs)

    # ========================================
    # 3. Set RHS expressions and measurements
    # ========================================

    # Generate expressions from CasADi model functions
    rhs = system.f(t, states, inputs)
    outputs = system.h(t, states, inputs)

    # Set righthand-side expressions for system states
    for i, name in enumerate(system.state_names):
        model.set_rhs(name, rhs[i])

    # Set righthand-side expressions for unmeasured disturbances
    for name in control_design.get('unmeasured_disturbances', []):
        model.set_rhs(
            name,
            cas.DM(0)  # d_dot = 0 + process_noise (added by estimator)
        )

    # Set righthand-side expressions for measured disturbances
    for name in control_design.get('measured_disturbances', []):
        model.set_rhs(
            name,
            cas.DM(0)  # d_dot = 0 (assumed constant or updated externally)
        )

    # Define measured variables and output expressions
    for name in control_design["controlled_variables"]:
        i = system.output_names.index(name)
        model.set_meas(meas_name=name, expr=outputs[i])

    # Setup model
    model.setup()

    # ========================================
    # 4. Rebuild expressions after model.setup()
    # ========================================
    # Re-build state vector after model.setup() called
    states = cas.vcat([model.x[name] for name in system.state_names])

    # Re-build input vector after model.setup() called
    inputs = []
    for name in system.input_names:
        if name in control_design['manipulated_variables']:
            inputs.append(model.u[name])
        elif name in control_design.get('measured_disturbances', []):
            inputs.append(model.x[name])
        elif name in control_design.get('unmeasured_disturbances', []):
            inputs.append(model.x[name])
    inputs = cas.vcat(inputs)

    # Re-generate expressions from CasADi model functions
    outputs = system.h(t, states, inputs)

    # ========================================
    # 5. Create MPC Controller
    # ========================================
    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(**mpc_params)

    # ========================================
    # 6. Define MPC Objective Function
    # ========================================

    # Build objective: sum of squared tracking errors
    mterm = cas.DM(0)  # Terminal cost (not used)
    lterm = cas.DM(0)  # Stage cost

    for cv_name, weight in cv_weights.items():
        if weight == 0:
            continue
        sp = setpoints[cv_name]
        cv_expr = outputs[system.output_names.index(cv_name)]
        error = cv_expr - sp
        lterm = lterm + weight * error ** 2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Penalize control effort (optional)
    if mv_weights is not None:
        rterm_dict = {
            name: mv_weights.get(name, 0.0)
            for name in control_design['manipulated_variables']
        }
        mpc.set_rterm(**rterm_dict)

    # ========================================
    # 7. Set MPC Constraints
    # ========================================

    # Input constraints
    if 'inputs' in bounds:
        for var_name, var_bounds in bounds['inputs'].items():
            if var_name in control_design['manipulated_variables']:
                if 'lower' in var_bounds:
                    mpc.bounds['lower', '_u', var_name] = var_bounds['lower']
                if 'upper' in var_bounds:
                    mpc.bounds['upper', '_u', var_name] = var_bounds['upper']

    # State constraints
    if 'states' in bounds:
        for var_name, var_bounds in bounds['states'].items():
            if var_name in system.state_names:
                if 'lower' in var_bounds:
                    mpc.bounds['lower', '_x', var_name] = var_bounds['lower']
                if 'upper' in var_bounds:
                    mpc.bounds['upper', '_x', var_name] = var_bounds['upper']

    # Disturbance bounds
    if 'disturbances' in bounds:
        for var_name, var_bounds in bounds['disturbances'].items():
            unmeas_dists = control_design.get('unmeasured_disturbances', [])
            meas_dists = control_design.get('measured_disturbances', [])
            if var_name in unmeas_dists or var_name in meas_dists:
                if 'lower' in var_bounds:
                    mpc.bounds['lower', '_x', var_name] = (
                        var_bounds['lower']
                    )
                if 'upper' in var_bounds:
                    mpc.bounds['upper', '_x', var_name] = (
                        var_bounds['upper']
                    )

    # Output constraints (using nonlinear constraints)
    if 'outputs' in bounds:
        for output_name, output_bounds in bounds['outputs'].items():
            if output_name not in system.output_names:
                raise ValueError(
                    f"Output '{output_name}' not found in "
                    f"system.output_names: {system.output_names}"
                )

            # Get the output expression
            output_idx = system.output_names.index(output_name)
            output_expr = outputs[output_idx]

            # Add upper bound constraint: y <= upper
            if 'upper' in output_bounds:
                mpc.set_nl_cons(
                    f'{output_name}_upper',
                    output_expr,
                    ub=output_bounds['upper']
                )

            # Add lower bound constraint: -y <= -lower (i.e., y >= lower)
            if 'lower' in output_bounds:
                mpc.set_nl_cons(
                    f'{output_name}_lower',
                    -output_expr,
                    ub=-output_bounds['lower']
                )

    # Setup MPC
    mpc.setup()

    return mpc, model
