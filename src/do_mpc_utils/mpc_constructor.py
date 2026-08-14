"""Functions for constructing MPC controllers from CasADi system models."""

import warnings

import casadi as cas
import do_mpc


def generate_system_f_and_h_as_symbolic_vectors(model, control_design, system):
    # Vector of symbolic state variables
    states = cas.vcat([model.x[name] for name in system.state_names])

    # Vector of symbolic input variables
    inputs = []
    for name in system.input_names:
        if name in control_design["manipulated_variables"]:
            inputs.append(model.u[name])
        elif name in control_design.get(
            "measured_disturbances", []
        ) or name in control_design.get("unmeasured_disturbances", []):
            inputs.append(model.x[name])
    inputs = cas.vcat(inputs)

    # Generate expressions using CasADi system model functions
    t = 0  # assume time invariant

    # Righthand-side of ODE equations
    rhs = system.f(t, states, inputs)

    # Output variables
    outputs = system.h(t, states, inputs)

    return rhs, outputs


def cost_function_setpoint_tracking(controlled_variables, setpoints, weights):
    pred_errors = setpoints - controlled_variables
    cost = cas.sum1(weights * cas.sumsqr(pred_errors))
    return cost


def construct_mpc(
    system,
    control_design,
    mpc_params,
    cv_weights,
    setpoints=None,
    mv_weights=None,
    bounds=None,
    model_type="continuous",
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
        - 'system_states': (optional) list of state names to include in MPC
            model. If not provided, all states from system.state_names are
            used.
        - 'measured_disturbances': (optional) list of measured disturbance
            input names. Must be a subset of system.input_names.
        - 'manipulated_variables': list of MV names. Must be a subset of
            system.input_names.
        - 'unmeasured_disturbances': (optional) list of unmeasured disturbance
            input names. Must be a subset of system.input_names. Must be a
            subset of system.output_names.
        - 'measured_outputs': (optional) list of output variable names. Must
            be a subset of system.output_names. If not provided, all system
            outputs are assumed to be measured unless listed in
            unmeasured_outputs.
        - 'unmeasured_outputs': (optional) list of output variable names.
            Must be a subset of system.output_names. If not provided, all
            system outputs not listed in measured_outputs will be included
            as unmeasured outputs.

    mpc_params : dict
        MPC setup parameters with keys, for example:
        - 't_step': time step for discretization (required)
        - 'n_horizon': prediction horizon steps (required)
        - 'n_robust': robust horizon (optional, default: 0)
        - 'store_full_solution': whether to store full solution
                   (optional, default: True)

    cv_weights : dict
        Dictionary of tracking error weights for controlled variables
        (CVs). Keys are variable names which may be from system.input_names,
        system.state_names, or system.output_names. Values are weights
        applied to the tracking errors in the objective function.

    setpoints : dict, optional
        Dictionary of setpoints for controlled variables.
        Keys are CV names. Values are either target values or the names
        of time-varying parameters to be used as setpoints. If None, all
        setpoints default to 0.

    mv_weights : dict, optional
        Dictionary of control effort weights for manipulated variables.
        Keys are MV names, values are weights for penalizing control
        effort. If None, no control effort penalty term is added to
        the cost function.

    bounds : dict, optional
        Dictionary of variable bounds with structure:
        {
            'system_states': {
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
    ...     'system_states': {
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

    # ========================================
    # 1. Validate control design parameters
    # ========================================

    # Get state names from control_design or use all system states
    state_names = control_design.get("system_states", system.state_names)

    # Validate control_design

    # Check no duplicates
    all_inputs = (
        control_design.get("manipulated_variables", [])
        + control_design.get("unmeasured_disturbances", [])
        + control_design.get("measured_disturbances", [])
    )
    if len(set(all_inputs)) != len(all_inputs):
        raise ValueError(
            "Duplicate variable name in manipulated_variables, "
            "unmeasured_disturbances or measured_disturbances."
        )

    # Check all input variable names are valid
    if set(all_inputs) != set(system.input_names):
        raise ValueError(
            f"Control design inputs {all_inputs} are not the "
            f"same as the system inputs {system.input_names}"
        )

    # Validate state names
    if set(state_names) != set(system.state_names):
        raise ValueError(
            f"Control design states {set(state_names)} do not match "
            f"system states {set(system.state_names)}. "
            f"Currently, all states must be included (reordering is allowed)."
        )
    # TODO: Support selecting a subset of states by creating dummy symbolic
    # values for non-selected states and checking if resulting expressions
    # contain any free variables

    # Validate controlled variable names
    for cv_name in cv_weights.keys():
        if (
            cv_name in system.output_names
            or cv_name in system.input_names
            or cv_name in system.state_names
        ):
            continue
        raise ValueError(
            f"Controlled variable {cv_name!r} not in system input_names,"
            "state_names or output_names."
        )

    # Set defaults for other optional arguments
    unmeasured_outputs = control_design.get("unmeasured_outputs", [])
    measured_outputs = control_design.get("measured_outputs", None)
    if measured_outputs is None:
        measured_outputs = set(cv_weights.keys()) - set(unmeasured_outputs)
    if bounds is None:
        bounds = {}
    if setpoints is None:
        setpoints = {cv: 0.0 for cv in cv_weights.keys()}
    else:
        # Warn if setpoints are specified without corresponding weights
        for sp_name, sp in setpoints.items():
            if sp_name not in cv_weights:
                if sp is None:
                    continue
                warnings.warn(
                    f"Setpoint specified for '{sp_name}' but no corresponding "
                    f"cv_weight. This setpoint will have no effect on the cost "
                    f"function.",
                    UserWarning,
                )

    # ========================================
    # 2. Create do-mpc model with variables
    # ========================================
    model = do_mpc.model.Model(model_type)

    # Add manipulated variables (MVs)
    for name in control_design["manipulated_variables"]:
        model.set_variable(var_type="_u", var_name=name, shape=(1, 1))

    # Add state variables (can be in any order specified by control_design)
    for name in state_names:
        model.set_variable(var_type="_x", var_name=name, shape=(1, 1))

    # Augment model with additional states for unmeasured disturbances
    for name in control_design.get("unmeasured_disturbances", []):
        model.set_variable(var_type="_x", var_name=name, shape=(1, 1))

    # Augment model with additional states for measured disturbances
    for name in control_design.get("measured_disturbances", []):
        model.set_variable(var_type="_x", var_name=name, shape=(1, 1))

    # Create time-varying parameters for time-varying setpoints
    for cv_name, sp_value in setpoints.items():
        # If value is a string, the setpoint is a time-varying parameter
        if isinstance(sp_value, str):
            model.set_variable(var_type="_tvp", var_name=sp_value, shape=(1, 1))

    # ========================================
    # 3. Define ODE and output expressions
    # ========================================

    # Build expressions for state and input vectors from CasADi system model
    rhs, outputs = generate_system_f_and_h_as_symbolic_vectors(
        model, control_design, system
    )

    # Set righthand-side expressions for system states
    for i, name in enumerate(system.state_names):
        model.set_rhs(name, rhs[i])

    # Set righthand-side expressions for unmeasured disturbances
    # TODO: Does this need to be generalised to any disturbance model?
    for name in control_design.get("unmeasured_disturbances", []):
        model.set_rhs(
            name,
            cas.DM(0),  # d_dot = 0 + process_noise (added by estimator)
        )

    # Set righthand-side expressions for measured disturbances
    # TODO: Replace this with TVP.
    for name in control_design.get("measured_disturbances", []):
        model.set_rhs(
            name,
            cas.DM(0),  # d_dot = 0 (assumed constant or updated externally)
        )

    # Define measured output variables
    for name in measured_outputs:
        i = system.output_names.index(name)
        model.set_meas(meas_name=name, expr=outputs[i])

    # Define unmeasured output variables as auxiliary expressions
    for name in unmeasured_outputs:
        i = system.output_names.index(name)
        model.set_expression(expr_name=name, expr=outputs[i])

    # Setup model
    model.setup()

    # ========================================
    # 4. Create MPC Controller
    # ========================================
    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(**mpc_params)

    # ========================================
    # 5. Define MPC Objective Function
    # ========================================

    weights = []
    expressions = []
    sp_values = []
    for name, weight in cv_weights.items():
        if weight == 0.0:
            continue
        sp_value = setpoints.get(name, 0.0)
        if isinstance(sp_value, str):
            # Setpoint is a time-varying parameter
            sp_value = model.tvp[sp_value]
        if name in model.y.keys():
            # CV is a measured output variable
            expr = model.y[name]
        elif name in model.u.keys():
            # CV is an MV
            expr = model.u[name]
        elif name in model.x.keys():
            # CV is a state variable
            expr = model.x[name]
        elif name in model.aux.keys():
            # CV is an auxiliary variable (e.g. unmeasured output)
            expr = model.aux[name]
        else:
            raise ValueError(
                f"Controlled variable {name!r} not in model.y, model.u,"
                "model.x, or model.aux."
            )
        weights.append(weight)
        expressions.append(expr)
        sp_values.append(sp_value)

        print(f"{name = }, {expr = }, {weight = }, {sp_value = }")

    # Sum-of-squared tracking errors
    lterm = cost_function_setpoint_tracking(
        cas.vcat(expressions), cas.vcat(sp_values), cas.vcat(weights)
    )
    print(f"{lterm = }")

    # Terminal cost
    mterm = cas.DM(0)

    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Set control action penalties (optional)
    if mv_weights is not None:
        rterm_dict = {
            name: mv_weights.get(name, 0.0)
            for name in control_design["manipulated_variables"]
        }
        mpc.set_rterm(**rterm_dict)

    # ========================================
    # 6. Set MPC Constraints
    # ========================================

    # Input constraints
    if "inputs" in bounds:
        for var_name, var_bounds in bounds["inputs"].items():
            if var_name in control_design["manipulated_variables"]:
                if "lower" in var_bounds:
                    mpc.bounds["lower", "_u", var_name] = var_bounds["lower"]
                if "upper" in var_bounds:
                    mpc.bounds["upper", "_u", var_name] = var_bounds["upper"]

    # State constraints
    if "system_states" in bounds:
        for var_name, var_bounds in bounds["system_states"].items():
            if var_name in system.state_names:
                if "lower" in var_bounds:
                    mpc.bounds["lower", "_x", var_name] = var_bounds["lower"]
                if "upper" in var_bounds:
                    mpc.bounds["upper", "_x", var_name] = var_bounds["upper"]

    # Disturbance bounds
    if "disturbances" in bounds:
        for var_name, var_bounds in bounds["disturbances"].items():
            unmeas_dists = control_design.get("unmeasured_disturbances", [])
            meas_dists = control_design.get("measured_disturbances", [])
            if var_name in unmeas_dists or var_name in meas_dists:
                if "lower" in var_bounds:
                    mpc.bounds["lower", "_x", var_name] = var_bounds["lower"]
                if "upper" in var_bounds:
                    mpc.bounds["upper", "_x", var_name] = var_bounds["upper"]

    # Output constraints (using nonlinear constraints)
    if "outputs" in bounds:
        for output_name, output_bounds in bounds["outputs"].items():
            if output_name not in system.output_names:
                raise ValueError(
                    f"Output '{output_name}' not found in "
                    f"system.output_names: {system.output_names}"
                )

            # Get the output expression
            output_idx = system.output_names.index(output_name)
            output_expr = outputs[output_idx]

            # Add upper bound constraint: y <= upper
            if "upper" in output_bounds:
                mpc.set_nl_cons(
                    f"{output_name}_upper",
                    output_expr,
                    ub=output_bounds["upper"],
                )

            # Add lower bound constraint: -y <= -lower (i.e., y >= lower)
            if "lower" in output_bounds:
                mpc.set_nl_cons(
                    f"{output_name}_lower",
                    -output_expr,
                    ub=-output_bounds["lower"],
                )

    # ========================================
    # 7. Finalize MPC setup
    # ========================================

    # Setup MPC
    mpc.setup()

    return mpc, model
