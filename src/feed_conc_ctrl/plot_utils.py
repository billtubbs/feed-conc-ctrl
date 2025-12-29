"""Plotting utilities for simulation results."""

import matplotlib.pyplot as plt


def make_tsplots(
    data, plot_info, units=None, time_units='hours', time_label='Time ({time_units})'
):
    """Create time series plots with multiple subplots.

    Args:
        data: DataFrame with time series data
        plot_info: Dictionary mapping subplot titles to {legend_name: column_name} dicts
        units: Dictionary mapping variable names to unit strings for y-axis labels
        time_units: String describing time units (default: 'hours')
        time_label: Format string for x-axis label (default: 'Time ({time_units})')

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
    n_subplots = len(plot_info)
    width, height = 8, 1 + 1.5*n_subplots

    fig, axes = plt.subplots(n_subplots, 1, sharex=True, figsize=(width, height))

    # Handle case of single subplot (axes is not a list)
    if n_subplots == 1:
        axes = [axes]

    for ax, (title, sub_plot_info) in zip(axes, plot_info.items()):
        for name, var_name in sub_plot_info.items():
            data[var_name].plot(ax=ax, label=name)

        # Set y-axis label from units dict if provided
        if units and var_name in units:
            ax.set_ylabel(units[var_name])

        ax.grid(True)
        ax.legend()
        ax.set_title(title)

    axes[-1].set_xlabel(time_label.format(time_units=time_units))

    return fig, axes
