"""Plotting utilities for simulation results."""

import matplotlib.pyplot as plt


def make_tsplots(
    data,
    plot_info=None,
    units=None,
    time_units=None,
    time_label=None,
    legend_loc="best",
    figsize=None,
    axes=None,
    **kwargs,
):
    """Create time series plots with multiple subplots.

    Args:
        data: DataFrame with time series data
        plot_info: Dictionary mapping subplot titles to {legend_name: column_name} dicts
        units: Dictionary mapping variable names to unit strings for y-axis labels
        time_units: String describing time units, e.g. 'hours', (default: None)
        time_label: Format string for x-axis label, e.g. 'Time ({time_units})', (default: None)
        legend_loc: Legend location (default: 'best'). Use 'outside right' to place
            legend to the right of the plot area, or any standard matplotlib location
            string ('upper left', 'lower right', etc.)
        axes: Optional array of existing Axes to plot onto. If provided, no new
            figure is created and figsize is ignored.

    Returns:
        fig, axes: Matplotlib figure and axes objects

    Example:
        >>> plot_info = {
        ...     "Tank Levels": {
        ...         "Tank 1": 'tank_1_L',
        ...         "Tank 2": 'tank_2_L'
        ...     },
        ...     "Concentrations": {
        ...         "Tank 1": 'tank_1_conc_out',
        ...         "Tank 2": 'tank_2_conc_out'
        ...     }
        ... }
        >>> units = {'tank_1_L': 'm', 'tank_2_L': 'm',
        ...          'tank_1_conc_out': 'kg/m³', 'tank_2_conc_out': 'kg/m³'}
        >>> fig, axes = make_tsplots(data, plot_info, units)
    """
    if plot_info is None:
        plot_info = {col: {col: {"var_name": col}} for col in data.columns}
    if data.index.name is not None:
        time_label = data.index.name
    if time_units is None:
        if time_label is None:
            time_label = "Time"
    else:
        if time_label is None:
            time_label = "Time {time_units}"

    n_subplots = len(plot_info)

    if axes is not None:
        # Use provided axes; derive figure from first axis
        if n_subplots == 1 and not hasattr(axes, '__len__'):
            axes = [axes]
        fig = axes[0].get_figure()
    else:
        if figsize is None:
            figsize = (8, 1.2 + 1.8 * n_subplots)
        fig, axes = plt.subplots(n_subplots, 1, sharex=True, figsize=figsize)
        # Handle case of single subplot (axes is not a list)
        if n_subplots == 1:
            axes = [axes]

    for ax, (title, sub_plot_info) in zip(axes, plot_info.items()):
        # Accumulate units for all variables in this subplot
        subplot_units = set()

        for name, info in sub_plot_info.items():
            info = info.copy()  # avoid modifying original
            var_name = info.pop("var_name")  # remove to avoid passing to plot
            try:
                kind = info.pop("kind")
            except KeyError:
                kind = "line"
            local_kwargs = kwargs.copy()
            local_kwargs.update(info)
            if kind == "line":
                data[var_name].plot(ax=ax, label=name, **local_kwargs)
            elif kind == "step":
                data[var_name].plot(
                    ax=ax, label=name, drawstyle="steps-post", **local_kwargs
                )
            else:
                raise ValueError(f"Unsupported plot kind: {kind}")

            # Collect units for this variable
            if units and var_name in units:
                unit = units[var_name]
                subplot_units.add(unit)

        # Set y-axis label from accumulated units
        if subplot_units:
            ax.set_ylabel(", ".join(subplot_units))
        else:
            ax.set_ylabel(", ".join(subplot_units))
        ax.grid(True)

        # Place legend based on legend_loc parameter
        if legend_loc == "outside right":
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(loc=legend_loc)

        ax.set_title(title)

    axes[-1].set_xlabel(time_label.format(time_units=time_units))

    return fig, axes
