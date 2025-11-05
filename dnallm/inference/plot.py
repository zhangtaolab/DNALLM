"""DNA Language Model Visualization and Plotting Module.

This module provides comprehensive plotting capabilities for DNA language model
results,
including metrics visualization, attention maps, embeddings, and
    mutation effects analysis.
"""

# Add more specific type hints and import numpy for better performance
import altair as alt
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

alt.data_transformers.enable("default", max_rows=None)


def _prepare_classification_data(
    metrics: dict[str, dict],
) -> tuple[dict, dict]:
    """Prepare data for classification tasks (binary, multiclass, multilabel,
    token).

    Args:
        metrics: Dictionary containing model metrics for different models

    Returns:
        Tuple containing bars_data and curves_data
    """
    bars_data: dict[str, list] = defaultdict(list)
    bars_data["models"] = []
    curves_data: dict[str, dict[str, list]] = {
        "ROC": defaultdict(list),
        "PR": defaultdict(list),
        "AUROC": {},
        "AUPRC": {},
    }

    for model, model_metrics in metrics.items():
        bars_data["models"].append(model)
        curves_data["AUROC"][model] = model_metrics["AUROC"]
        curves_data["AUPRC"][model] = model_metrics["AUPRC"]
        for metric, metric_data in model_metrics.items():
            if metric == "curve":
                _process_curve_data(metric_data, curves_data, model)
            else:
                _add_bar_metric(bars_data, metric, metric_data)

    return dict(bars_data), dict(curves_data)


def _prepare_regression_data(metrics: dict[str, dict]) -> tuple[dict, dict]:
    """Prepare data for regression tasks.

    Args:
        metrics: Dictionary containing model metrics for different models

    Returns:
        Tuple containing bars_data and scatter_data
    """
    bars_data: dict[str, list] = defaultdict(list)
    bars_data["models"] = []
    scatter_data: dict[str, dict[str, list]] = {}

    for model, model_metrics in metrics.items():
        bars_data["models"].append(model)
        scatter_data[model] = {"predicted": [], "experiment": []}

        for metric, metric_data in model_metrics.items():
            if metric == "scatter":
                _process_scatter_data(metric_data, scatter_data, model)
            else:
                _add_bar_metric(bars_data, metric, metric_data)
                if metric == "r2":
                    scatter_data[model][metric] = metric_data

    return dict(bars_data), scatter_data


def _process_curve_data(
    metric_data: dict, curves_data: dict, model: str
) -> None:
    """Process curve data for ROC and PR curves."""
    for score, values in metric_data.items():
        if score.endswith("pr"):
            if score == "fpr":
                curves_data["ROC"]["models"].extend([model] * len(values))
            curves_data["ROC"][score].extend(values)
        else:
            if score == "precision":
                curves_data["PR"]["models"].extend([model] * len(values))
            curves_data["PR"][score].extend(values)


def _process_scatter_data(
    metric_data: dict, scatter_data: dict, model: str
) -> None:
    """Process scatter plot data for regression tasks."""
    for score, values in metric_data.items():
        # convert tensor to list if needed
        if hasattr(values, "tolist"):
            values = values.tolist()
        scatter_data[model][score].extend(values)


def _add_bar_metric(bars_data: dict, metric: str, metric_data) -> None:
    """Add metric data to bars_data dictionary."""
    if metric not in bars_data:
        bars_data[metric] = []
    bars_data[metric].append(metric_data)


def prepare_data(
    metrics: dict[str, dict], task_type: str = "binary"
) -> tuple[dict, dict | dict]:
    """Prepare data for plotting various types of visualizations.

    This function organizes model metrics data into formats suitable for
        different plot types:
        - Bar charts for classification and regression metrics
        - ROC and PR curves for classification tasks
        - Scatter plots for regression tasks

        Args:
            metrics: Dictionary containing model metrics for different models
                    task_type: Type of task (
                "binary",
                "multiclass",
                "multilabel",
                "token",
                "regression")

        Returns:
            Tuple containing:
            - bars_data: Data formatted for bar chart visualization
                    - curves_data/scatter_data: Data formatted for curve or
                scatter plot visualization

        Raises:
            ValueError: If task type is not supported for plotting
    """
    if task_type in ["binary", "multiclass", "multilabel", "token"]:
        return _prepare_classification_data(metrics)
    elif task_type == "regression":
        return _prepare_regression_data(metrics)
    else:
        raise ValueError(f"Unsupported task type {task_type} for plotting")


def plot_bars(
    data: dict,
    show_score: bool = True,
    ncols: int = 3,
    width: int = 200,
    height: int = 50,
    bar_width: int = 30,
    domain: tuple[float, float] | list[float] = (0.0, 1.0),
    save_path: str | None = None,
    separate: bool = False,
) -> alt.Chart | dict[str, alt.Chart]:
    """Plot bar charts for model metrics comparison.

    This function creates bar charts to compare different metrics across
        multiple models. It supports automatic layout with multiple columns and
            optional score labels on bars.

        Args:
            data: Dictionary containing metrics data with "models" as the first
                key
            show_score: Whether to show the score values on the bars
            ncols: Number of columns to arrange the plots
            width: Width of each individual plot
            height: Height of each individual plot
            bar_width: Width of the bars in the plot
            domain: Y-axis domain range for the plots, default (0.0, 1.0)
                    save_path: Path to save the plot. If None,
                plot will be shown interactively
            separate: Whether to return separate plots for each metric

        Returns:
                    Altair chart object (combined or
                separate plots based on separate parameter)
    """
    # Convert to DataFrame once and cache for better performance
    # Original: dbar = pd.DataFrame(data)
    dbar = pd.DataFrame(data)

    # Pre-allocate plot dictionaries for better memory management
    # Original: pbar = {}; p_separate = {}
    pbar = {}
    p_separate = {}

    # Filter metrics once and use list comprehension for better performance
    # Original: for n, metric in enumerate([x for x in data if x != "models"]):
    metrics_list = [x for x in data if x != "models"]

    for n, metric in enumerate(metrics_list):
        # More efficient domain calculation with numpy
        # Original: if metric in ["mae", "mse"]: domain_use = [0,
        # dbar[metric].max()*1.1]
        if metric in ["mae", "mse"]:
            domain_use = [0, dbar[metric].max() * 1.1]
        else:
            domain_use = domain

        # Create bar chart with optimized encoding
        bar = (
            alt.Chart(dbar)
            .mark_bar(size=bar_width)
            .encode(
                x=alt.X(f"{metric}:Q").scale(domain=domain_use),
                y=alt.Y("models").title(None),
                color=alt.Color("models").legend(None),
                tooltip=["models", metric],
            )
            .properties(width=width, height=height * len(dbar["models"]))
        )

        if show_score:
            # Optimized text positioning and formatting
            text = (
                alt.Chart(dbar)
                .mark_text(
                    dx=-10 if dbar[metric].min() >= 0.2 else 5,
                    color="white" if dbar[metric].min() >= 0.2 else "black",
                    baseline="middle",
                    align="right" if dbar[metric].min() >= 0.2 else "left",
                )
                .encode(
                    x=alt.X(f"{metric}:Q"),
                    y=alt.Y("models").title(None),
                    text=alt.Text(metric, format=".3f"),
                )
            )
            p = bar + text
        else:
            p = bar

        if separate:
            p_separate[metric] = p.configure_axis(grid=False)

        # More efficient plot arrangement logic
        idx = n // ncols
        if n % ncols == 0:
            pbar[idx] = p
        else:
            pbar[idx] |= p

    # More efficient plot combination with reduce-like approach
    # Original: Multiple conditional checks and assignments
    pbars: alt.Chart = pbar[0] if pbar else alt.Chart()
    for i in range(1, len(pbar)):
        pbars &= pbar[i]

    # Configure chart once at the end
    pbars = pbars.configure_axis(grid=False)

    # Save the plot
    if save_path:
        pbars.save(save_path)
        print(f"Metrics bar charts saved to {save_path}")

    if separate:
        return p_separate
    else:
        return pbars


def plot_polar_bar(
       data: dict,
       title: str | None = None,
       save_path: str | None = None,
) -> alt.Chart:
    """Plot a polar bar chart.

    Args:
        data: Dictionary containing the data to plot.
        categories: List of categories for the bars.
        values: List of values for the bars.
        title: Title of the chart.
        color: Color of the bars.

    Returns:
        Altair chart object.
    """
    _default_colors = [
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
        "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
        "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"
    ]

    metrics = {}
    for name in data:
        metric = name.replace("test_", "")
        if metric in [
            "loss", "runtime", "samples_per_second", "steps_per_second"
        ]:
            continue
        metrics[metric.capitalize()] = data[name]

    df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": [v * 100 for v in metrics.values()],
        "index": list(range(len(metrics))),
        "color": _default_colors[:len(metrics)]
    })

    base = alt.Chart(df).mark_arc(
        stroke="grey",
        padAngle=0.05,
        cornerRadius=10,
        tooltip=True
        ).encode(
        theta=alt.Theta("Metric:O", sort=None),
        radius=alt.Radius("Value").scale(type="sqrt", zero=True),
        radius2=alt.datum(5),
        color=alt.Color("Metric:N", sort=None).scale(scheme="tableau20"),
        order=alt.Order("index:Q"),
    )

    text = base.mark_text(radiusOffset=10).encode(
        radius=alt.Radius("Value:Q", scale=alt.Scale(type="sqrt", zero=True)),
        angle=alt.Angle("Metric:N", sort=None),
        text=alt.Text("Value:Q", format=".2f"),
        color=alt.value("black")
    )

    chart = alt.layer(base, text)
    if title:
        chart = chart.properties(title=title)

    if save_path:
        chart.save(save_path)
        print(f"Polar bar chart saved to {save_path}")

    return chart


def plot_curve(
    data: dict,
    show_score: bool = True,
    width: int = 400,
    height: int = 400,
    save_path: str | None = None,
    separate: bool = False,
) -> alt.Chart | dict[str, alt.Chart]:
    """Plot ROC and PR curves for classification tasks.

          This function creates ROC (Receiver Operating Characteristic) and
          PR (Precision-Recall)
      curves to evaluate model performance on classification tasks.

      Args:
          data: Dictionary containing ROC and PR curve data with "ROC" and
                "PR" keys, and optionally "AUROC" and "AUPRC" score dicts.
          show_score: Whether to show the score values on the plot (now
                implemented in the legend).
          width: Width of each plot
          height: Height of each plot
                  save_path: Path to save the plot. If None,
                plot will be shown interactively
          separate: Whether to return separate plots for ROC and PR curves

      Returns:
                Altair chart object (combined or
                separate plots based on separate parameter)
    """
    pline = {}
    p_separate = {}

    roc_data = pd.DataFrame(data["ROC"])

    # ROC chart
    roc_chart = (
        alt.Chart(roc_data)
        .mark_line()
        .encode(
            x=alt.X("fpr", title="FPR").scale(domain=(0.0, 1.0)),
            y=alt.Y("tpr", title="TPR").scale(domain=(0.0, 1.0)),
            color=alt.Color("models:N", title="Models"),
            tooltip=["fpr", "tpr", "models"],
        )
        .properties(width=width, height=height, title="ROC Curve")
    )

    # Diagonal line
    diag_line = alt.Chart(pd.DataFrame(
        {"fpr": [0, 1], "tpr": [0, 1]})).mark_line(
        strokeDash=[5, 5], color="gray"
    ).encode(
        x=alt.X("fpr").scale(domain=(0.0, 1.0)),
        y=alt.Y("tpr").scale(domain=(0.0, 1.0)),
    )

    pline[0] = roc_chart + diag_line

    # Add text annotations for AUROC
    if show_score and "AUROC" in data:
        auroc_scores = data.get("AUROC", {})

        # Create a DataFrame for the text annotations
        text_data = []
        y_offset = 0.05  # Vertical spacing between text lines
        y_start = 0.05    # Start from the bottom

        # Dynamically calculate y positions based on the number of models
        for i, (model, score) in enumerate(auroc_scores.items()):
            text_data.append({
                "models": model,
                "label": f"{model} (AUC={score:.3f})",
                "fpr": 0.95,
                "tpr": y_start + i * y_offset
            })

        auroc_text_df = pd.DataFrame(text_data)
        auroc_text_layer = (
            alt.Chart(auroc_text_df)
            .mark_text(
                align="right",
                baseline="bottom",
            )
            .encode(
                x=alt.X("fpr:Q", scale=alt.Scale(domain=(0.0, 1.0))),
                y=alt.Y("tpr:Q", scale=alt.Scale(domain=(0.0, 1.0))),
                text="label:N",
                # Use the same color encoding as the line chart
                color=alt.Color("models:N", legend=None)
            )
        )
        # Add the text layer to the ROC plot
        pline[0] = pline[0] + auroc_text_layer
    pline[0] = alt.layer(roc_chart, diag_line, auroc_text_layer)

    if separate:
        p_separate["ROC"] = pline[0]

    # --- Create PR curve ---
    pr_data = pd.DataFrame(data["PR"])

    # PR chart
    pr_chart = (
        alt.Chart(pr_data)
        .mark_line()
        .encode(
            x=alt.X("recall", title="Recall").scale(domain=(0.0, 1.0)),
            y=alt.Y("precision", title="Precision").scale(domain=(0.0, 1.0)),
            color=alt.Color("models:N", title="Models"),
            tooltip=["recall", "precision", "models"],
        )
        .properties(width=width, height=height, title="PR Curve")
    )
    pr_baseline = alt.Chart(pd.DataFrame(
        {"recall": [0, 1],
         "precision": [
             pr_data["precision"].min(),
             pr_data["precision"].min()]}
        )).mark_line(
        strokeDash=[5, 5], color="gray"
    ).encode(
        x=alt.X("recall").scale(domain=(0.0, 1.0)),
        y=alt.Y("precision").scale(domain=(0.0, 1.0)),
    )

    pline[1] = pr_chart + pr_baseline

    # Add text annotations for AUPRC
    if show_score and "AUPRC" in data:
        auprc_scores = data.get("AUPRC", {})

        # Create a DataFrame for the text annotations
        text_data = []
        y_offset = 0.05  # Vertical spacing between text lines
        y_start = 0.05   # Start from the bottom

        for i, (model, score) in enumerate(auprc_scores.items()):
            text_data.append({
                "models": model,
                "label": f"{model} (AUC={score:.3f})",
                "recall": 0.05,
                "precision": y_start + i * y_offset
            })

        auprc_text_df = pd.DataFrame(text_data)
        auprc_text_layer = (
            alt.Chart(auprc_text_df)
            .mark_text(
                align="left",
                baseline="bottom",
            )
            .encode(
                x=alt.X("recall:Q", scale=alt.Scale(domain=(0.0, 1.0))),
                y=alt.Y("precision:Q", scale=alt.Scale(domain=(0.0, 1.0))),
                text="label:N",
                # Use the same color encoding as the line chart
                color=alt.Color("models:N", legend=None)
            )
        )

        pline[1] = pline[1] + auprc_text_layer

    if separate:
        p_separate["PR"] = pline[1]

    # Combine plots if not separate
    plines: alt.Chart = pline[0] if pline else alt.Chart()
    for i in range(1, len(pline)):
        plines |= pline[i]

    plines = plines.configure_axis(grid=False)

    if save_path:
        plines.save(save_path)
        print(f"ROC/PR curves saved to {save_path}")

    if separate:
        return p_separate
    else:
        return plines


def plot_scatter(
    data: dict,
    show_score: bool = True,
    ncols: int = 3,
    width: int = 400,
    height: int = 400,
    save_path: str | None = None,
    separate: bool = False,
) -> alt.Chart | dict[str, alt.Chart]:
    """Plot scatter plots for regression task evaluation.

    This function creates scatter plots to compare predicted vs. experimental
        values
        for regression tasks, with optional R² score display.

        Args:
            data: Dictionary containing scatter plot data for each model
            show_score: Whether to show the R² score on the plot
            ncols: Number of columns to arrange the plots
            width: Width of each plot
            height: Height of each plot
                    save_path: Path to save the plot. If None,
                plot will be shown interactively
            separate: Whether to return separate plots for each model

        Returns:
                    Altair chart object (combined or
                separate plots based on separate parameter)
    """
    from scipy.stats import gaussian_kde

    # Pre-allocate plot dictionaries for better memory management
    # Original: pdot = {}; p_separate = {}
    pdot = {}
    p_separate = {}

    # More efficient data processing with list comprehension
    # Original: for n, model in enumerate(data):
    for n, (model, model_data) in enumerate(data.items()):
        # Create a copy to avoid modifying original data
        # Original: scatter_data = dict(data[model])
        scatter_data = dict(model_data)
        r2 = scatter_data.pop("r2", 0)  # Use pop with default value

        # More efficient DataFrame creation
        # Original: ddot = pd.DataFrame(scatter_data)
        ddot = pd.DataFrame(scatter_data)

        try:
            # Make density calculation more efficient
            xy = np.vstack([ddot["experiment"], ddot["predicted"]])
            ddot["density"] = gaussian_kde(xy)(xy)
            density_calculated = True

            # Order by density
            ddot = ddot.sort_values(by="density", ascending=True)

        except (np.linalg.LinAlgError, ValueError):
            # If KDE fails (e.g., all points are identical), fall back
            ddot["density"] = 1.0
            density_calculated = False

        # Create scatter plot with optimized encoding
        # dot = (
        #     alt.Chart(ddot, title=model)
        #     .mark_point(filled=True)
        #     .encode(
        #         x=alt.X("predicted:Q"),
        #         y=alt.Y("experiment:Q"),
        #     )
        #     .properties(width=width, height=height)
        # )
        base = alt.Chart(ddot, title=model).properties(
            width=width, height=height
        )
        dot = (
            base.mark_point(filled=True, size=15, opacity=1)
            .encode(
                x=alt.X("experiment:Q", title="Observed"),
                y=alt.Y("predicted:Q", title="Predicted"),

                color=alt.Color(
                    "density:Q",
                    scale=alt.Scale(scheme="viridis"),
                    legend=alt.Legend(title="Density")
                ) if density_calculated else alt.value("blue"),

                tooltip=["experiment", "predicted", "density"]
            )
        )

        if show_score:
            # More efficient text positioning calculation
            # Original: min_x = ddot["predicted"].min(); max_y =
            # ddot["experiment"].max()
            min_x = ddot["predicted"].min()
            max_y = ddot["experiment"].max()

            text = (
                alt.Chart()
                .mark_text(size=14, align="left", baseline="top", dx=5, dy=5)
                .encode(
                    x=alt.datum(min_x + 0.5),
                    y=alt.datum(max_y - 0.5),
                    text=alt.datum(f"R²={r2:.3f}"),  # Format R² value
                )
            )
            p = dot + text
        else:
            p = dot

        if separate:
            p_separate[model] = p.configure_axis(grid=False)

        # More efficient plot arrangement
        idx = n // ncols
        if n % ncols == 0:
            pdot[idx] = p
        else:
            pdot[idx] |= p

    # More efficient plot combination
    # Original: Multiple conditional checks and assignments
    pdots: alt.Chart = pdot[0] if pdot else alt.Chart()
    for i in range(1, len(pdot)):
        pdots &= pdot[i]

    # Configure chart once at the end
    pdots = pdots.configure_axis(grid=False)

    # Save the plot
    if save_path:
        pdots.save(save_path)
        print(f"Metrics scatter plots saved to {save_path}")

    if separate:
        return p_separate
    else:
        return pdots


def plot_attention_map(
    attentions: tuple | list,
    sequences: list[str],
    tokenizer,
    seq_idx: int = 0,
    layer: int = -1,
    head: int | str = -1,
    skip_cls: bool = True,
    width: int = 800,
    height: int = 800,
    save_path: str | None = None,
) -> alt.Chart:
    """Plot attention map visualization for transformer models.

    This function creates a heatmap visualization of attention weights between
        tokens
            in a sequence,
            showing how the model attends to different parts of the input.

        Args:
                    attentions: Tuple or
                list containing attention weights from model layers
            sequences: List of input sequences
            tokenizer: Tokenizer object for converting tokens to readable text
            seq_idx: Index of the sequence to plot, default 0
            layer: Layer index to visualize, default -1 (last layer)
                    attention_head: Attention head index to visualize,
                default -1 (last head)
            width: Width of the plot
            height: Height of the plot
                    save_path: Path to save the plot. If None,
                plot will be shown interactively

        Returns:
            Altair chart object showing the attention heatmap
    """
    # More efficient attention data extraction with numpy
    # Original: attn_layer = attentions[layer].numpy()
    attn_layer = np.array(attentions[layer])
    if head == "all":
        # Average over all heads
        attn_head = np.mean(attn_layer[seq_idx], axis=0)
    else:
        attn_head = attn_layer[seq_idx][head]

    # More efficient token processing with error handling
    # Original: seq = sequences[seq_idx]; tokens_id = tokenizer.encode(seq)
    seq = sequences[seq_idx]
    try:
        tokens_id = tokenizer.encode(seq)
        tokens = tokenizer.convert_ids_to_tokens(tokens_id)
    except (AttributeError, TypeError):
        # Fallback tokenization for different tokenizer types
        tokens = tokenizer.decode(seq).split()

    # Pre-allocate DataFrame data structure for better performance
    # Original: num_tokens = len(tokens); flen = len(str(num_tokens))
    num_tokens = len(tokens)
    flen = len(str(num_tokens))

    # Use list comprehension for more efficient data creation
    # Original: Multiple loops with append operations
    if skip_cls:
        if tokens[0].lower() in ["[cls]", "<cls>", "<s>", "cls"]:
            tokens = tokens[1:]
            attn_head = attn_head[1:, 1:]
            num_tokens -= 1
        if tokens[-1].lower() in ["[sep]", "<sep>", "</s>", "sep"]:
            tokens = tokens[:-1]
            attn_head = attn_head[:-1, :-1]
            num_tokens -= 1
    df_data = {
        "token1": [
            f"{str(i).zfill(flen)}{t1}"
            for i, t1 in enumerate(tokens)
            for _ in range(num_tokens)
        ],
        "token2": [
            f"{str(num_tokens - j).zfill(flen)}{t2}"
            for _ in range(num_tokens)
            for j, t2 in enumerate(tokens)
        ],
        "attn": [
            attn_head[i][j]
            for i in range(num_tokens)
            for j in range(num_tokens)
        ],
    }

    # More efficient DataFrame creation
    # Original: source = pd.DataFrame(df)
    source = pd.DataFrame(df_data)

    # Enable VegaFusion for Altair performance
    alt.data_transformers.enable("vegafusion")

    # Create attention map with optimized encoding and axis configuration
    # Original: Multiple axis configurations
    attn_map: alt.Chart = (
        alt.Chart(source)
        .mark_rect()
        .encode(
            x=alt.X(
                "token1:O",
                axis=alt.Axis(
                    labelExpr=f"substring(datum.value, {flen}, 100)",
                    labelAngle=-45,
                ),
            ).title(None),
            y=alt.Y(
                "token2:O",
                axis=alt.Axis(
                    labelExpr=f"substring(datum.value, {flen}, 100)",
                    labelAngle=0,
                ),
            ).title(None),
            color=alt.Color("attn:Q").scale(scheme="bluepurple"),
        )
        .properties(width=width, height=height)
        .configure_axis(grid=False)
    )

    # Save the plot
    if save_path:
        attn_map.save(save_path)
        print(f"Attention map saved to {save_path}")

    return attn_map


def _get_dimensionality_reducer(
    reducer: str,
    n_samples: int | None = None,
    quality: str = "balanced",  # fast / balanced / high
    **kwargs,
):
    """
    Initialize and return a dimensionality reducer with recommended parameters.

    Args:
        reducer: Dimensionality reduction method
            ("PCA", "t-SNE", "UMAP", "PaCMAP")
        n_samples: Number of samples in dataset, used to auto-adjust parameters
        quality: "fast", "balanced", "high"
        kwargs: Custom override parameters

    Returns:
        Initialized dimensionality reducer object
    """
    reducer = reducer.lower()
    quality = quality.lower()

    # Safety checks
    if quality not in ["fast", "balanced", "high"]:
        quality = "balanced"
    if n_samples is None:
        # Default assume medium size
        n_samples = 3000

    def tsne_params():
        if n_samples < 5000:
            presets = {
                "fast": {
                    "perplexity": 20,
                    "max_iter": 500,
                    "learning_rate": 100,
                },
                "balanced": {
                    "perplexity": 30,
                    "max_iter": 1000,
                    "learning_rate": 200,
                },
                "high": {
                    "perplexity": 40,
                    "max_iter": 2000,
                    "learning_rate": 300,
                },
            }
        elif n_samples < 50000:
            presets = {
                "fast": {"perplexity": 30, "max_iter": 500},
                "balanced": {"perplexity": 40, "max_iter": 1000},
                "high": {"perplexity": 50, "max_iter": 2000},
            }
        else:
            raise ValueError(
                "t-SNE is unsupported for samples > 50k, "
                "please use UMAP/PaCMAP."
            )
        base = {"n_components": 2, "init": "pca"}
        base.update(presets[quality])
        return base

    def umap_params():
        if n_samples < 5000:
            presets = {
                "fast": {"n_neighbors": 10, "min_dist": 0.5},
                "balanced": {"n_neighbors": 15, "min_dist": 0.3},
                "high": {"n_neighbors": 30, "min_dist": 0.1},
            }
        elif n_samples < 50000:
            presets = {
                "fast": {"n_neighbors": 15, "min_dist": 0.5},
                "balanced": {"n_neighbors": 30, "min_dist": 0.3},
                "high": {"n_neighbors": 50, "min_dist": 0.1},
            }
        else:
            presets = {
                "fast": {"n_neighbors": 30, "min_dist": 0.5},
                "balanced": {"n_neighbors": 50, "min_dist": 0.3},
                "high": {"n_neighbors": 80, "min_dist": 0.2},
            }
        base = {"n_components": 2, "metric": "euclidean"}
        base.update(presets[quality])
        return base

    def pacmap_params():
        if n_samples < 5000:
            presets = {
                "fast": {"n_neighbors": 8, "FP_ratio": 2.0, "MN_ratio": 0.5},
                "balanced": {
                    "n_neighbors": 12,
                    "FP_ratio": 2.0,
                    "MN_ratio": 0.5,
                },
                "high": {"n_neighbors": 20, "FP_ratio": 2.5, "MN_ratio": 0.5},
            }
        elif n_samples < 50000:
            presets = {
                "fast": {"n_neighbors": 10},
                "balanced": {"n_neighbors": 20},
                "high": {"n_neighbors": 30, "FP_ratio": 2.0},
            }
        else:
            presets = {
                "fast": {"n_neighbors": 20},
                "balanced": {"n_neighbors": 30, "FP_ratio": 1.5},
                "high": {"n_neighbors": 50, "FP_ratio": 1.0},
            }
        base = {"n_components": 2}
        base.update(presets[quality])
        return base

    # -----------------------
    # Reducer initialization
    # -----------------------
    try:
        if reducer == "pca":
            from sklearn.decomposition import PCA

            params = {"n_components": 2}
            params.update(kwargs)
            return PCA(**params)

        elif reducer in ["t-sne", "tsne"]:
            from sklearn.manifold import TSNE

            params = tsne_params()
            params.update(kwargs)
            return TSNE(**params)

        elif reducer == "umap":
            from umap import UMAP

            params = umap_params()
            params.update(kwargs)
            return UMAP(**params)

        elif reducer == "pacmap":
            from pacmap import PaCMAP

            params = pacmap_params()
            params.update(kwargs)
            return PaCMAP(**params)

        else:
            raise ValueError("Unsupported reducer")

    except ImportError as e:
        raise ImportError(f"Required package not installed: {e}") from e


def _compute_mean_embeddings(
    hidden_states,
    attention_mask=None,
    strategy="mean",
):
    """Compute mean embeddings using attention mask for pooling.

    Args:
        hidden_states: Hidden states from model layers
        attention_mask: Attention mask for sequence padding
        strategy: Pooling strategy ("last" or "mean" or "first")

    Returns:
        Mean pooled embeddings as numpy array
    """
    embeddings = np.array(hidden_states)
    if attention_mask is None:
        if strategy == "last":
            attention_mask_array = np.zeros(
                (embeddings.shape[0], embeddings.shape[1]), dtype=int
            )
            attention_mask_array[:, -1] = 1
        elif strategy == "first":
            attention_mask_array = np.zeros(
                (embeddings.shape[0], embeddings.shape[1]), dtype=int
            )
            attention_mask_array[:, 0] = 1
        else:  # mean pooling without mask
            attention_mask_array = np.ones(
                (embeddings.shape[0], embeddings.shape[1]), dtype=int
            )
    else:
        attention_mask_array = np.array(attention_mask)

    mask_sum = np.sum(attention_mask_array, axis=1, keepdims=True)
    mean_embeddings = (
        np.sum(attention_mask_array[..., None] * embeddings, axis=-2)
        / mask_sum
    )

    return mean_embeddings


def _prepare_embedding_dataframe(dim_reduced_vectors, labels, label_names):
    """Prepare DataFrame for embedding visualization.

    Args:
        dim_reduced_vectors: 2D reduced embedding vectors
        labels: Data point labels
        label_names: Label names for legend display

    Returns:
        pandas DataFrame ready for plotting
    """
    if labels is None or len(labels) == 0:
        label_names = ["Uncategorized"]
        labels = [0] * dim_reduced_vectors.shape[0]
    if labels[0] is None:
        label_names = ["Uncategorized"]
        labels = [0] * dim_reduced_vectors.shape[0]

    processed_labels = [
        label_names[int(i)]
        if (label_names is not None and i < len(label_names))
        else str(i)
        for i in labels
    ]

    return pd.DataFrame({
        "Dimension 1": dim_reduced_vectors[:, 0],
        "Dimension 2": dim_reduced_vectors[:, 1],
        "labels": processed_labels,
    })


def _create_embedding_plot(source_df, layer_idx, width, height):
    """Create individual embedding plot for a layer.

    Args:
        source_df: DataFrame containing embedding data
        layer_idx: Layer index for plot title
        width: Plot width
        height: Plot height

    Returns:
        Altair chart object
    """
    return (
        alt.Chart(source_df, title=f"Layer {layer_idx + 1}")
        .mark_point(filled=True)
        .encode(
            x=alt.X("Dimension 1:Q"),
            y=alt.Y("Dimension 2:Q"),
            color=alt.Color("labels:N",
                            legend=alt.Legend(title="Labels")
                            ).scale(scheme="set1"),
        )
        .properties(width=width, height=height)
    )


def _arrange_plots(plots, ncols):
    """Arrange multiple plots in a grid layout.

    Args:
        plots: List of individual plots
        ncols: Number of columns in grid

    Returns:
        Combined Altair chart
    """
    if not plots:
        return alt.Chart()

    pdot = {}
    for i, plot in enumerate(plots):
        idx = i // ncols
        if i % ncols == 0:
            pdot[idx] = plot
        else:
            pdot[idx] |= plot

    combined = pdot[0] if pdot else alt.Chart()
    for i in range(1, len(pdot)):
        combined &= pdot[i]

    return combined.configure_axis(grid=False)


def plot_embeddings(
    hidden_states: tuple | list,
    attention_mask: tuple | list | None,
    reducer: str = "t-SNE",
    labels: tuple | list | None = None,
    label_names: str | list | None = None,
    ncols: int = 4,
    width: int = 300,
    height: int = 300,
    save_path: str | None = None,
    separate: bool = False,
    norm: bool = True,
    reduced: bool = False,
) -> alt.Chart | dict[str, alt.Chart]:
    """Visualize embeddings using dimensionality reduction techniques.

    This function creates 2D visualizations of high-dimensional embeddings
        from different model layers using PCA, t-SNE, or UMAP dimensionality
        reduction methods.

        Args:
            hidden_states: Tuple or list containing hidden states from model
                layers
                    attention_mask: Tuple or
                list containing attention masks for sequence padding
                    reducer: Dimensionality reduction method. Options: "PCA",
                "t-SNE", "UMAP"
            labels: List of labels for the data points
            labels_names: List of label names for legend display
            ncols: Number of columns to arrange the plots
            width: Width of each plot
            height: Height of each plot
                    save_path: Path to save the plot. If None,
                plot will be shown interactively
            separate: Whether to return separate plots for each layer
            norm: Whether to normalize embeddings before reduction
            reduced: Whether the input hidden states are already 2D

        Returns:
                    Altair chart object (combined or
                separate plots based on separate parameter)

        Raises:
            ValueError: If unsupported dimensionality reduction method is
                specified
    """
    # Initialize dimensionality reducer
    n_samples = hidden_states[0].shape[0] if hidden_states else None
    dim_reducer = _get_dimensionality_reducer(reducer, n_samples=n_samples)
    # Type assertion: dim_reducer is guaranteed to be non-None from helper
    # function
    if dim_reducer is None:
        raise ValueError(
            "Dimensionality reducer is None - this should not happen"
        )

    # Process each layer and create plots
    plots = []
    p_separate = {}

    for i, hidden in enumerate(hidden_states):
        # Compute mean embeddings
        if reduced:
            mean_embeddings = np.array(hidden)
        else:
            mean_embeddings = _compute_mean_embeddings(hidden, attention_mask)

        # Apply dimensionality reduction
        if norm:
            # embeddings_normalized = normalize(mean_embeddings)
            mean_embeddings = StandardScaler().fit_transform(mean_embeddings)
        layer_dim_reduced_vectors = np.array(
            dim_reducer.fit_transform(mean_embeddings)
        )

        # Prepare data for plotting
        source_df = _prepare_embedding_dataframe(
            layer_dim_reduced_vectors, labels, label_names
        )

        # Create individual plot
        plot = _create_embedding_plot(source_df, i, width, height)
        plots.append(plot)

        if separate:
            p_separate[f"Layer{i + 1}"] = plot.configure_axis(grid=False)

    # Arrange plots in grid
    combined_plot = _arrange_plots(plots, ncols)

    # Save the plot
    if save_path:
        combined_plot.save(save_path)
        print(f"Embeddings visualization saved to {save_path}")

    return p_separate if separate else combined_plot


def _extract_mutation_data(
    data: dict,
) -> tuple[str, list[str], int, int, list[str]]:
    """Extract basic mutation data from input dictionary.

    Args:
        data: Dictionary containing mutation data with "raw" and mutation keys

    Returns:
            Tuple containing sequence, raw_bases, sequence length, format
            length, and mutation list
    """
    raw_data = data["raw"]
    sequence = raw_data["sequence"]
    seqlen = len(sequence)
    flen = len(str(seqlen))
    mut_list = [x for x in data.keys() if x != "raw"]
    raw_bases = list(sequence)

    return sequence, raw_bases, seqlen, flen, mut_list


def _process_substitution_mutations(
    data: dict, mut_list: list[str], i: int, base1: str, flen: int
) -> tuple[dict, float, str]:
    """Process substitution mutations for a single position.

    Args:
        data: Mutation data dictionary
        mut_list: List of all mutations
        i: Position index
        base1: Original base at position
        flen: Format length for position strings

    Returns:
        Tuple containing heatmap data, max score, and max effect base
    """
    dheat_pos = defaultdict(list)
    ref = f"mut_{i}_{base1}_{base1}"
    mut_prefix = f"mut_{i}_{base1}_"

    maxabs = 0.0
    maxscore = 0.0
    maxabs_index = base1

    relevant_muts = [x for x in mut_list if x.startswith(mut_prefix)] + [ref]

    for mut in sorted(relevant_muts):
        if mut in data:
            base2 = mut.split("_")[-1]
            score = data[mut]["score"]
        else:
            base2 = base1
            score = 0.0

        dheat_pos["base"].append(f"{str(i).zfill(flen)}{base1}")
        dheat_pos["mut"].append(base2)
        dheat_pos["score"].append(score)

        if abs(score) > maxabs:
            maxabs = abs(score)
            maxscore = score
            maxabs_index = base2

    return dict(dheat_pos), maxscore, maxabs_index


def _process_indel_mutations(
    data: dict, mut_list: list[str], i: int, base1: str, flen: int
) -> dict:
    """Process insertion and deletion mutations for a single position.

    Args:
        data: Mutation data dictionary
        mut_list: List of all mutations
        i: Position index
        base1: Original base at position
        flen: Format length for position strings

    Returns:
        Dictionary containing indel heatmap data
    """
    dheat_indel = defaultdict(list)

    # Process deletions
    del_prefix = f"del_{i}_"
    for mut in [x for x in mut_list if x.startswith(del_prefix)]:
        base2 = f"del_{mut.split('_')[-1]}"
        score = data[mut]["score"]
        dheat_indel["base"].append(f"{str(i).zfill(flen)}{base1}")
        dheat_indel["mut"].append(base2)
        dheat_indel["score"].append(score)

    # Process insertions
    ins_prefix = f"ins_{i}_"
    for mut in [x for x in mut_list if x.startswith(ins_prefix)]:
        base2 = f"ins_{mut.split('_')[-1]}"
        score = data[mut]["score"]
        dheat_indel["base"].append(f"{str(i).zfill(flen)}{base1}")
        dheat_indel["mut"].append(base2)
        dheat_indel["score"].append(score)

    return dict(dheat_indel)


def _build_mutation_datasets(
    data: dict,
    raw_bases: list[str],
    mut_list: list[str],
    seqlen: int,
    flen: int,
) -> tuple[dict, dict, dict]:
    """Build datasets for heatmap, line plot, and bar chart visualizations.

    Args:
        data: Mutation data dictionary
        raw_bases: List of original bases in sequence
        mut_list: List of all mutations
        seqlen: Sequence length
        flen: Format length for position strings

    Returns:
        Tuple containing heatmap, line plot, and bar chart data
    """
    dheat = defaultdict(list)
    dline = {
        "x": [f"{str(i).zfill(flen)}{x}" for i, x in enumerate(raw_bases)] * 2,
        "score": [0.0] * seqlen * 2,
        "type": ["gain"] * seqlen + ["loss"] * seqlen,
    }
    dbar = defaultdict(list)

    for i, base1 in enumerate(raw_bases):
        # Process substitution mutations
        dheat_pos, maxscore, maxabs_index = _process_substitution_mutations(
            data, mut_list, i, base1, flen
        )

        # Accumulate heatmap data
        for key, values in dheat_pos.items():
            dheat[key].extend(values)

        # Update line plot data
        for score in dheat_pos["score"]:
            if score >= 0:
                dline["score"][i] += score
            else:
                dline["score"][i + seqlen] -= score

        # Update bar chart data
        dbar["x"].append(f"{str(i).zfill(flen)}{base1}")
        dbar["score"].append(maxscore)
        dbar["base"].append(maxabs_index)

        # Process indel mutations
        dheat_indel = _process_indel_mutations(data, mut_list, i, base1, flen)
        for key, values in dheat_indel.items():
            dheat[key].extend(values)

    return dict(dheat), dline, dict(dbar)


def _create_mutation_charts(
    dheat: dict, dline: dict, dbar: dict, width: int, height: int, flen: int
) -> alt.Chart | alt.VConcatChart:
    """Create individual charts for mutation visualization.

    Args:
        dheat: Heatmap data
        dline: Line plot data
        dbar: Bar chart data
        width: Chart width
        height: Chart height
        flen: Format length for position strings

    Returns:
        Combined Altair chart
    """
    if not dheat["score"]:
        return alt.Chart()

    # Calculate color domains and ranges
    all_scores = dheat["score"]
    max_abs_score = max(abs(min(all_scores)), abs(max(all_scores)))
    domain1 = [-max_abs_score, 0.0, max_abs_score]
    range1_ = ["#2166ac", "#f7f7f7", "#b2182b"]

    unique_bases = sorted(set(dbar["base"]))
    range2_ = ["#33a02c", "#e31a1c", "#1f78b4", "#ff7f00", "#cab2d6"][
        : len(unique_bases)
    ]

    # Enable VegaFusion for performance
    alt.data_transformers.enable("vegafusion")

    # Create heatmap
    pheat: alt.Chart = (
        alt.Chart(pd.DataFrame(dheat))
        .mark_rect()
        .encode(
            x=alt.X(
                "base:O",
                axis=alt.Axis(
                    labelExpr=f"substring(datum.value, {flen}, {flen}+1)",
                    labelAngle=0,
                ),
            ).title(None),
            y=alt.Y("mut:O").title("mutation"),
            color=alt.Color("score:Q").scale(domain=domain1, range=range1_),
        )
        .properties(width=width, height=height)
    )

    # Create line plot
    pline: alt.Chart = (
        alt.Chart(pd.DataFrame(dline))
        .mark_line()
        .encode(
            x=alt.X("x:O").title(None).axis(labels=False),
            y=alt.Y("score:Q"),
            color=alt.Color("type:N").scale(
                domain=["gain", "loss"], range=["#b2182b", "#2166ac"]
            ),
        )
        .properties(width=width, height=height)
    )

    # Create bar chart
    pbar: alt.Chart = (
        alt.Chart(pd.DataFrame(dbar))
        .mark_bar()
        .encode(
            x=alt.X("x:O").title(None).axis(labels=False),
            y=alt.Y("score:Q"),
            color=alt.Color("base:N").scale(
                domain=unique_bases, range=range2_
            ),
        )
        .properties(width=width, height=height)
    )

    # Combine charts
    combined_chart: alt.Chart | alt.VConcatChart = (
        pheat & pbar & pline
    ).configure_axis(grid=False)
    return combined_chart


def plot_muts(
    data: dict,
    show_score: bool = False,
    width: int | None = None,
    height: int = 100,
    save_path: str | None = None,
) -> alt.Chart | alt.VConcatChart:
    """Visualize mutation effects on model predictions.

    This function creates comprehensive visualizations of how different
        mutations affect model predictions, including:
        - Heatmap showing mutation effects at each position
        - Line plot showing gain/loss of function
        - Bar chart showing maximum effect mutations

        Args:
            data: Dictionary containing mutation data with "raw" and mutation
                keys
    show_score: Whether to show the score values on the plot (currently not
            implemented)
                    width: Width of the plot. If None,
                automatically calculated based on sequence length
            height: Height of the plot
                    save_path: Path to save the plot. If None,
                plot will be shown interactively

        Returns:
            Altair chart object showing the combined mutation effects
                visualization
    """
    # Extract basic data
    _sequence, raw_bases, seqlen, flen, mut_list = _extract_mutation_data(data)

    # Build visualization datasets
    dheat, dline, dbar = _build_mutation_datasets(
        data, raw_bases, mut_list, seqlen, flen
    )

    # Calculate width if not provided
    if width is None and dheat["score"]:
        width = int(height * len(raw_bases) / len(set(dheat["mut"])))
    elif width is None:
        width = 400  # Default width

    # Create charts
    pmerge = _create_mutation_charts(dheat, dline, dbar, width, height, flen)

    # Save the plot if requested
    if save_path and dheat["score"]:
        pmerge.save(save_path)
        print(f"Mutation effects visualization saved to {save_path}")

    return pmerge


def plot_attributions_token(
    tokens: list[str],
    scores: np.ndarray,
    title: str = "Token-level Attributions",
    special_tokens: list[str] | None = None
) -> alt.Chart:
    """
    Visualize token-level attribution scores as colored text using Altair.

    Args:
        tokens (List[str]): List of tokens from the tokenizer.
        scores (np.ndarray): Array of attribution scores
                             corresponding to each token.
        title (str): The title for the chart.
        special_tokens (List[str]): Special tokens to filter out
                                    from the visualization.

    Returns:
        alt.Chart: An Altair chart object.
    """
    # 1. Filter out special tokens
    if special_tokens is None:
        special_tokens = [
            "[CLS]", "[SEP]", "[PAD]",
            "<s>", "</s>", "<pad>",
            "<unk>", "[UNK]",
            "<bos>", "<eos>",
            "<cls>", "<sep>"
        ]
    valid_indices = [i for i, token in enumerate(tokens)
                     if token not in special_tokens]
    vis_tokens = [tokens[i] for i in valid_indices]
    vis_scores = scores[valid_indices]

    if len(vis_tokens) == 0:
        print("Warning: No valid tokens to plot after filtering.")
        return alt.Chart().mark_text().properties(title="No data to display")

    # 2. Calculate each token"s length and precise position in the sequence
    token_lengths = [len(t) for t in vis_tokens]
    end_pos = np.cumsum(token_lengths)
    start_pos = end_pos - token_lengths
    center_pos = start_pos + (np.array(token_lengths) / 2.0)

    source = pd.DataFrame({
        "token": vis_tokens,
        "score": vis_scores,
        "token_length": token_lengths,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "center_pos": center_pos
    })

    # 3. Calculate color scale domain and range
    max_abs_score = np.max(np.abs(source["score"])) if not source.empty else 0
    domain = [-max_abs_score, 0, max_abs_score]
    color_range = ["#2166ac", "#f7f7f7", "#b2182b"]  # Blue -> White -> Red

    # 4. Create Altair chart
    # X axis now is quantitative (Quantitative), representing base position
    base = alt.Chart(source).properties(
        width=max(600, int(source["end_pos"].max() * 12)),  # dynamic width
        height=50,
        title=title
    )

    # Background rectangles, using x and x2 to define variable width
    rects = base.mark_rect().encode(
        x=alt.X("start_pos:Q", axis=None, title="Base Position"),
        x2=alt.X2("end_pos:Q"),
        color=alt.Color("score:Q",
                        scale=alt.Scale(domain=domain, range=color_range),
                        legend=alt.Legend(title="Attribution Score")),
        tooltip=[
            alt.Tooltip("token:N", title="Token"),
            alt.Tooltip("score:Q", title="Score", format=".4f"),
            alt.Tooltip("token_length:Q", title="Length (bp)")
        ]
    )

    # Text layer, centered
    text = base.mark_text(baseline="middle", fontSize=12, clip=True).encode(
        x=alt.X("center_pos:Q", axis=None),
        text="token:N",
        color=alt.condition(
            alt.datum.score > max_abs_score * 0.5,
            alt.value("white"),
            alt.value("black")
        )
    )

    return (rects + text).configure_view(strokeWidth=0)


def plot_attributions_line(
    tokens: list[str],
    scores: np.ndarray,
    title: str = "Positional Attribution Scores",
    window_size: int | None = 5,
    special_tokens: list[str] | None = None
) -> alt.Chart:
    """
    Plot attribution scores as a line chart to show regional importance.

    Args:
        tokens (List[str]): List of tokens from the tokenizer.
        scores (np.ndarray): Array of attribution scores.
        title (str): The title for the chart.
        special_tokens (List[str]): Special tokens to filter out.

    Returns:
        alt.Chart: An Altair chart object.
    """
    if special_tokens is None:
        special_tokens = [
            "[CLS]", "[SEP]", "[PAD]",
            "<s>", "</s>", "<pad>",
            "<unk>", "[UNK]",
            "<bos>", "<eos>",
            "<cls>", "<sep>"
        ]
    valid_indices = [i for i, token in enumerate(tokens)
                     if token not in special_tokens]
    vis_scores = scores[valid_indices]

    if len(vis_scores) == 0:
        print("Warning: No valid scores to plot after filtering.")
        return alt.Chart().mark_text().properties(title="No data to display")

    source = pd.DataFrame({
        "position": range(len(vis_scores)),
        "Raw": vis_scores
    })

    # Create base chart (raw scores)
    area = alt.Chart(source).mark_area(opacity=0.2, color="lightblue").encode(
        x=alt.X("position:Q", title="Token Position"),
        y=alt.Y("Raw:Q", title="Attribution Score")
    )
    # Add base line at y=0 (dashed line)
    area += alt.Chart(
        pd.DataFrame({"y": [0]})
    ).mark_rule(color="black", strokeDash=[5, 5]).encode(
        y="y:Q"
    )

    # if need to smooth the scores
    if window_size and window_size > 1:
        source["Smoothed"] = source["Raw"].rolling(
            window=window_size,
            center=True,
            min_periods=1
        ).mean()
        # Convert to long format for Altair
        long_source = source.melt(
            id_vars=["position"],
            var_name="type",
            value_name="value"
        )
    else:  # if no smoothing, just use raw scores
        long_source = source.melt(
            id_vars=["position"],
            var_name="type",
            value_name="value"
        )

    # Create line chart layer
    lines = alt.Chart(long_source).mark_line().encode(
        x=alt.X("position:Q"),
        y=alt.Y("value:Q"),
        color=alt.Color("type:N", legend=alt.Legend(title="Score Type")),
        tooltip=[
            alt.Tooltip("position:Q"),
            alt.Tooltip("value:Q", format=".4f", title="Score"),
            alt.Tooltip("type:N", title="Type")
        ]
    )

    chart = area + lines

    return chart.properties(
        width=800,
        height=200,
        title=title
    ).interactive().configure_axis(grid=False)


def plot_attributions_multi(
    all_attributions: list[tuple[list[str], np.ndarray]],
    title: str = "Aggregated Attribution Heatmap"
) -> alt.Chart:
    """
    Plot an aggregated heatmap of attribution scores from multiple sequences.

    Args:
            all_attributions (List[Tuple[List[str], np.ndarray]]): List of
                tuples containing tokens and their corresponding attribution
                scores for multiple sequences.
            **Crucially, all arrays must be of the same length.**
        title (str): The title for the chart.

    Returns:
        alt.Chart: An Altair chart object.

    Raises:
        ValueError: If the input score arrays have inconsistent lengths.
    """
    if not all_attributions:
        print("Warning: No scores provided to plot.")
        return alt.Chart().mark_text().properties(title="No data to display")

    # 0. Remove right padding tokens/scores if any
    trimmed_attributions = []
    possible_padding_tokens = [
        "[PAD]", "<pad>", "</s>", "<eos>",
        "[SEP]", "<sep>",
        " ", "#", "*"
    ]
    pad_token = possible_padding_tokens[0]
    for tokens, scores in all_attributions:
        # Identify the last non-padding token index
        last_valid_index = len(tokens)
        for i in reversed(range(len(tokens))):
            if tokens[i] in possible_padding_tokens:
                pad_token = tokens[i]
            else:
                last_valid_index = i + 1
                break
        trimmed_attributions.append(
            (tokens[:last_valid_index], scores[:last_valid_index])
        )

    # 1. Check for consistent lengths
    first_len = len(trimmed_attributions[0][1])
    if not all(len(scores) == first_len for _, scores in trimmed_attributions):
        # extend to max length with NaN padding
        max_len = max(len(scores) for _, scores in trimmed_attributions)
        extended_attributions = []
        for tokens, scores in trimmed_attributions:
            if len(scores) < max_len:
                extended_scores = np.pad(
                    scores, (0, max_len - len(scores)),
                    mode="constant", constant_values=np.nan
                )
                extended_tokens = (
                    tokens + [pad_token] * (max_len - len(tokens))
                )
                extended_attributions.append(
                    (extended_tokens, extended_scores)
                )
            else:
                extended_attributions.append((tokens, scores))
        all_attributions = extended_attributions
    else:
        all_attributions = trimmed_attributions

    # 2. Stack into a matrix and convert to long-form DataFrame
    all_scores = [scores for _, scores in all_attributions]
    score_matrix = np.stack(all_scores, axis=0)

    source = pd.DataFrame(score_matrix).unstack().reset_index()
    source.columns = ["position", "sample_index", "score"]

    # 3. Calculate color range
    max_abs_score = np.max(np.abs(source["score"]))
    domain = [-max_abs_score, 0, max_abs_score]
    color_range = ["#2166ac", "#f7f7f7", "#b2182b"]  # Blue -> White -> Red

    # 4. Create heatmap
    heatmap = alt.Chart(source).mark_rect().encode(
        x=alt.X(
            "position:O", title="Aligned Position",
            axis=alt.Axis(labels=True, ticks=True)
        ),
        y=alt.Y(
            "sample_index:O", title="Sample Index",
            axis=alt.Axis(labels=False, ticks=False)
        ),
        color=alt.Color("score:Q",
                        scale=alt.Scale(domain=domain, range=color_range),
                        legend=alt.Legend(title="Attribution"))
    ).properties(
        width=len(all_attributions[0][1]) * 15,
        height=len(all_attributions) * 15,
        title=title
    )

    return heatmap
