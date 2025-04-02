from typing import Union
import altair as alt
import pandas as pd


def prepare_data(metrics: dict, task_type: str="binary") -> tuple:
    """
    Prepare data for plotting.
    Args:
        metrics (dict): Dictionary containing model metrics.
        task_type (str): Task type
    Returns:
        tuple: Tuple containing bar data and curve data.
    """
    # Load the data
    bars_data = {'models': []}
    if task_type in ['binary', 'multiclass', 'multilabel']:
        curves_data = {'ROC': {'models': [], 'fpr': [], 'tpr': []},
                    'PR': {'models': [], 'recall': [], 'precision': []}}
        for model in metrics:
            if model not in bars_data['models']:
                bars_data['models'].append(model)
            for metric in metrics[model]:
                if metric == 'curve':
                    for score in metrics[model][metric]:
                        if score.endswith('pr'):
                            if score == 'fpr':
                                curves_data['ROC']['models'].extend([model] * len(metrics[model][metric][score]))
                            curves_data['ROC'][score].extend(metrics[model][metric][score])
                        else:
                            if score == 'precision':
                                curves_data['PR']['models'].extend([model] * len(metrics[model][metric][score]))
                            curves_data['PR'][score].extend(metrics[model][metric][score])
                else:
                    if metric not in bars_data:
                        bars_data[metric] = []
                    bars_data[metric].append(metrics[model][metric])
        return bars_data, curves_data
    elif task_type == "regression":
        scatter_data = {}
        for model in metrics:
            if model not in bars_data['models']:
                bars_data['models'].append(model)
            scatter_data[model] = {'predicted': [], 'experiment': []}
            for metric in metrics[model]:
                if metric == 'scatter':
                    for score in metrics[model][metric]:
                        scatter_data[model][score].extend(metrics[model][metric][score])
                else:
                    if metric not in bars_data:
                        bars_data[metric] = []
                    bars_data[metric].append(metrics[model][metric])
                    if metric == 'r2':
                        scatter_data[model][metric] = metrics[model][metric]
        return bars_data, scatter_data
    else:
        raise ValueError(f"Unsupport task type {task_type} for ploting")


def plot_bars(data: dict, show_score: bool = True, ncol: int = 3,
              width: int = 200, height: int = 50, bar_width: int = 30,
              domain: Union[tuple, list]= (0.0, 1.0),
              save_path: str = None) -> alt.Chart:
    """
    Plot bar chart.
    Args:
        data (dict): Data to be plotted.
        show_score (bool): Whether to show the score on the plot.
        ncol (int): Number of columns in the plot.
        width (int): Width of the plot.
        height (int): Height of the plot.
        bar_width (int): Width of the bars in the plot.
        save_path (str): Path to save the plot.
    Returns:
        pbars: Altair chart object.
    """
    # Plot bar charts
    dbar = pd.DataFrame(data)
    pbar = {}
    for n, metric in enumerate([x for x in data if x != 'models']):
        if metric in ['mae', 'mse']:
            domain_use = [0, dbar[metric].max()*1.1]
        else:
            domain_use = domain
        bar = alt.Chart(dbar).mark_bar(size=bar_width).encode(
            x=alt.X(metric + ":Q").scale(domain=domain_use),
            y=alt.Y("models").title(None),
            color=alt.Color('models').legend(None),
        ).properties(width=width, height=height*len(dbar['models']))
        if show_score:
            text = alt.Chart(dbar).mark_text(
                dx=-10,
                color='white',
                baseline='middle',
                align='right').encode(
                    x=alt.X(metric + ":Q"),
                    y=alt.Y("models").title(None),
                    text=alt.Text(metric, format='.3f')
                    )
            p = bar + text
        else:
            p = bar
        idx = n // ncol
        if n % ncol == 0:
            pbar[idx] = p
        else:
            pbar[idx] |= p
    # Combine the plots
    for i, p in enumerate(pbar):
        if i == 0:
            pbars = pbar[i]
        else:
            pbars &= pbar[i]
    # Configure the chart
    pbars = pbars.configure_axis(grid=False)
    # Save the plot
    if save_path:
        pbars.save(save_path)
        print(f"Metrics bar charts saved to {save_path}")
    return pbars


def plot_curve(data: dict, show_score: bool = True,
               width: int = 400, height: int = 400,
               save_path: str = None) -> alt.Chart:
    """
    Plot curve chart.
    Args:
        data (dict): Data to be plotted.
        show_score (bool): Whether to show the score on the plot.
        width (int): Width of the plot.
        height (int): Height of the plot.
        save_path (str): Path to save the plot.
    Returns:
        plines: Altair chart object.
    """
    # Plot curves
    pline = {}
    # Plot ROC curve
    roc_data = pd.DataFrame(data['ROC'])
    pline[0] = alt.Chart(roc_data).mark_line().encode(
        x=alt.X("fpr").scale(domain=(0.0, 1.0)),
        y=alt.Y("tpr").scale(domain=(0.0, 1.0)),
        color="models",
    ).properties(width=width, height=height)
    # Plot PR curve
    pr_data = pd.DataFrame(data['PR'])
    pline[1] = alt.Chart(pr_data).mark_line().encode(
        x=alt.X("recall").scale(domain=(0.0, 1.0)),
        y=alt.Y("precision").scale(domain=(0.0, 1.0)),
        color="models",
    ).properties(width=width, height=height)
    # Combine the plots
    for i, p in enumerate(pline):
        if i == 0:
            plines = pline[i]
        else:
            plines |= pline[i]
    # Configure the chart
    plines = plines.configure_axis(grid=False)
    # Save the plot
    if save_path:
        plines.save(save_path)
        print(f"ROC curves saved to {save_path}")
    return plines


def plot_scatter(data: dict, show_score: bool = True, ncol: int = 3,
                 width: int = 400, height: int = 400,
                 save_path: str = None) -> alt.Chart:
    """
    Plot scatter chart.
    Args:
        data (dict): Data to be plotted.
        show_score (bool): Whether to show the score on the plot.
        width (int): Width of the plot.
        height (int): Height of the plot.
        save_path (str): Path to save the plot.
    Returns:
        pdots: Altair chart object.
    """
    # Plot bar charts
    pdot = {}
    for n, model in enumerate(data):
        scatter_data = dict(data[model])
        r2 = scatter_data['r2']
        del scatter_data['r2']
        ddot = pd.DataFrame(scatter_data)
        dot = alt.Chart(ddot, title=model).mark_point(filled=True).encode(
            x=alt.X("predicted:Q"),
            y=alt.Y("experiment:Q"),
        ).properties(width=width, height=height)
        if show_score:
            min_x = ddot['predicted'].min()
            max_y = ddot['experiment'].max()
            text = alt.Chart().mark_text(size=14, align="left", baseline="bottom").encode(
                x=alt.datum(min_x + 0.5),
                y=alt.datum(max_y - 0.5),
                text=alt.datum("R\u00b2=" + str(r2))
            )
            p = dot + text
        else:
            p = dot
        idx = n // ncol
        if n % ncol == 0:
            pdot[idx] = p
        else:
            pdot[idx] |= p
    # Combine the plots
    for i, p in enumerate(pdot):
        if i == 0:
            pdots = pdot[i]
        else:
            pdots &= pdot[i]
    # Configure the chart
    pdots = pdots.configure_axis(grid=False)
    # Save the plot
    if save_path:
        pdots.save(save_path)
        print(f"Metrics scatter plots saved to {save_path}")
    return pdots
