"""Plotting utilities for simulation results."""

import matplotlib.pyplot as plt


def make_tsplots(
    data,
    plot_info,
    units=None,
    time_units="hours",
    time_label="Time ({time_units})",
    legend_loc="best",
):
    """Create time series plots with multiple subplots.

    Args:
        data: DataFrame with time series data
        plot_info: Dictionary mapping subplot titles to {legend_name: column_name} dicts
        units: Dictionary mapping variable names to unit strings for y-axis labels
        time_units: String describing time units (default: 'hours')
        time_label: Format string for x-axis label (default: 'Time ({time_units})')
        legend_loc: Legend location (default: 'best'). Use 'outside right' to place
            legend to the right of the plot area, or any standard matplotlib location
            string ('upper left', 'lower right', etc.)

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
        >>> # Or with legend outside:
        >>> fig, axes = make_tsplots(data, plot_info, units, legend_loc='outside right')
    """
    n_subplots = len(plot_info)
    width, height = 8, 1 + 1.5 * n_subplots

    fig, axes = plt.subplots(
        n_subplots, 1, sharex=True, figsize=(width, height)
    )

    # Handle case of single subplot (axes is not a list)
    if n_subplots == 1:
        axes = [axes]

    for ax, (title, sub_plot_info) in zip(axes, plot_info.items()):
        # Accumulate units for all variables in this subplot
        subplot_units = []

        for name, info in sub_plot_info.items():
            var_name = info["var_name"]
            kind = info.get("kind", "line")
            if kind == "line":
                data[var_name].plot(ax=ax, label=name)
            elif kind == "step":
                data[var_name].plot(ax=ax, label=name, drawstyle="steps-post")
            else:
                raise ValueError(f"Unsupported plot kind: {kind}")

            # Collect units for this variable
            if units and var_name in units:
                unit = units[var_name]
                if unit not in subplot_units:
                    subplot_units.append(unit)

        # Set y-axis label from accumulated units
        if subplot_units:
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
