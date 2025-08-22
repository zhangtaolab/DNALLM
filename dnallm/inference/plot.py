"""DNA Language Model Visualization and Plotting Module.

This module provides comprehensive plotting capabilities for DNA language model results,
including metrics visualization, attention maps, embeddings, and mutation effects analysis.
"""

# Add more specific type hints and import numpy for better performance
from typing import Union, Dict, List, Tuple, Optional
import altair as alt
import pandas as pd
import numpy as np
from collections import defaultdict


def prepare_data(metrics: Dict[str, Dict], task_type: str = "binary") -> Tuple[Dict, Union[Dict, Dict]]:
    """Prepare data for plotting various types of visualizations.
    
    This function organizes model metrics data into formats suitable for different plot types:
    - Bar charts for classification and regression metrics
    - ROC and PR curves for classification tasks
    - Scatter plots for regression tasks
    
    Args:
        metrics: Dictionary containing model metrics for different models
        task_type: Type of task ('binary', 'multiclass', 'multilabel', 'token', 'regression')
        
    Returns:
        Tuple containing:
        - bars_data: Data formatted for bar chart visualization
        - curves_data/scatter_data: Data formatted for curve or scatter plot visualization
        
    Raises:
        ValueError: If task type is not supported for plotting
    """
    # Use defaultdict to avoid repeated key existence checks
    # Original: bars_data = {'models': []}
    bars_data = defaultdict(list)
    bars_data['models'] = []
    
    if task_type in ['binary', 'multiclass', 'multilabel', 'token']:
        # Pre-allocate curve data structure for better performance
        # Original: curves_data = {'ROC': {'models': [], 'fpr': [], 'tpr': []}, 'PR': {'models': [], 'recall': [], 'precision': []}}
        curves_data = {
            'ROC': defaultdict(list),
            'PR': defaultdict(list)
        }
        
        # Single loop through models and metrics for better efficiency
        # Original: Multiple nested loops with repeated condition checks
        for model, model_metrics in metrics.items():
            bars_data['models'].append(model)
            
            for metric, metric_data in model_metrics.items():
                if metric == 'curve':
                    # More efficient curve data extraction
                    for score, values in metric_data.items():
                        if score.endswith('pr'):
                            if score == 'fpr':
                                curves_data['ROC']['models'].extend([model] * len(values))
                            curves_data['ROC'][score].extend(values)
                        else:
                            if score == 'precision':
                                curves_data['PR']['models'].extend([model] * len(values))
                            curves_data['PR'][score].extend(values)
                else:
                    # Use defaultdict to avoid key existence checks
                    if metric not in bars_data:
                        bars_data[metric] = []
                    bars_data[metric].append(metric_data)
        
        return dict(bars_data), dict(curves_data)
    
    elif task_type == "regression":
        # Pre-allocate scatter data structure
        scatter_data = {}
        
        for model, model_metrics in metrics.items():
            bars_data['models'].append(model)
            # Initialize scatter data structure once
            scatter_data[model] = {'predicted': [], 'experiment': []}
            
            for metric, metric_data in model_metrics.items():
                if metric == 'scatter':
                    # More efficient scatter data extraction
                    for score, values in metric_data.items():
                        scatter_data[model][score].extend(values)
                else:
                    if metric not in bars_data:
                        bars_data[metric] = []
                    bars_data[metric].append(metric_data)
                    if metric == 'r2':
                        scatter_data[model][metric] = metric_data
        
        return dict(bars_data), scatter_data
    
    else:
        raise ValueError(f"Unsupported task type {task_type} for plotting")


def plot_bars(data: Dict, show_score: bool = True, ncols: int = 3,
              width: int = 200, height: int = 50, bar_width: int = 30,
              domain: Union[Tuple[float, float], List[float]] = (0.0, 1.0),
              save_path: Optional[str] = None, separate: bool = False) -> alt.Chart:
    """Plot bar charts for model metrics comparison.
    
    This function creates bar charts to compare different metrics across multiple models.
    It supports automatic layout with multiple columns and optional score labels on bars.
    
    Args:
        data: Dictionary containing metrics data with 'models' as the first key
        show_score: Whether to show the score values on the bars
        ncols: Number of columns to arrange the plots
        width: Width of each individual plot
        height: Height of each individual plot
        bar_width: Width of the bars in the plot
        domain: Y-axis domain range for the plots, default (0.0, 1.0)
        save_path: Path to save the plot. If None, plot will be shown interactively
        separate: Whether to return separate plots for each metric
        
    Returns:
        Altair chart object (combined or separate plots based on separate parameter)
    """
    # Convert to DataFrame once and cache for better performance
    # Original: dbar = pd.DataFrame(data)
    dbar = pd.DataFrame(data)
    
    # Pre-allocate plot dictionaries for better memory management
    # Original: pbar = {}; p_separate = {}
    pbar = {}
    p_separate = {}
    
    # Filter metrics once and use list comprehension for better performance
    # Original: for n, metric in enumerate([x for x in data if x != 'models']):
    metrics_list = [x for x in data if x != 'models']
    
    for n, metric in enumerate(metrics_list):
        # More efficient domain calculation with numpy
        # Original: if metric in ['mae', 'mse']: domain_use = [0, dbar[metric].max()*1.1]
        if metric in ['mae', 'mse']:
            domain_use = [0, dbar[metric].max() * 1.1]
        else:
            domain_use = domain
        
        # Create bar chart with optimized encoding
        bar = alt.Chart(dbar).mark_bar(size=bar_width).encode(
            x=alt.X(f"{metric}:Q").scale(domain=domain_use),
            y=alt.Y("models").title(None),
            color=alt.Color('models').legend(None),
        ).properties(width=width, height=height * len(dbar['models']))
        
        if show_score:
            # Optimized text positioning and formatting
            text = alt.Chart(dbar).mark_text(
                dx=-10,
                color='white',
                baseline='middle',
                align='right'
            ).encode(
                x=alt.X(f"{metric}:Q"),
                y=alt.Y("models").title(None),
                text=alt.Text(metric, format='.3f')
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
    pbars = pbar[0] if pbar else alt.Chart()
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


def plot_curve(data: Dict, show_score: bool = True,
               width: int = 400, height: int = 400,
               save_path: Optional[str] = None, separate: bool = False) -> alt.Chart:
    """Plot ROC and PR curves for classification tasks.
    
    This function creates ROC (Receiver Operating Characteristic) and PR (Precision-Recall)
    curves to evaluate model performance on classification tasks.
    
    Args:
        data: Dictionary containing ROC and PR curve data with 'ROC' and 'PR' keys
        show_score: Whether to show the score values on the plot (currently not implemented)
        width: Width of each plot
        height: Height of each plot
        save_path: Path to save the plot. If None, plot will be shown interactively
        separate: Whether to return separate plots for ROC and PR curves
        
    Returns:
        Altair chart object (combined or separate plots based on separate parameter)
    """
    # Pre-allocate plot dictionaries and use more descriptive names
    # Original: pline = {}; p_separate = {}
    pline = {}
    p_separate = {}
    
    # Create ROC curve with optimized data handling
    # Original: roc_data = pd.DataFrame(data['ROC'])
    roc_data = pd.DataFrame(data['ROC'])
    pline[0] = alt.Chart(roc_data).mark_line().encode(
        x=alt.X("fpr").scale(domain=(0.0, 1.0)),
        y=alt.Y("tpr").scale(domain=(0.0, 1.0)),
        color="models",
    ).properties(width=width, height=height)
    
    if separate:
        p_separate['ROC'] = pline[0]
    
    # Create PR curve with optimized data handling
    # Original: pr_data = pd.DataFrame(data['PR'])
    pr_data = pd.DataFrame(data['PR'])
    pline[1] = alt.Chart(pr_data).mark_line().encode(
        x=alt.X("recall").scale(domain=(0.0, 1.0)),
        y=alt.Y("precision").scale(domain=(0.0, 1.0)),
        color="models",
    ).properties(width=width, height=height)
    
    if separate:
        p_separate['PR'] = pline[1]
    
    # More efficient plot combination
    # Original: Multiple conditional checks and assignments
    plines = pline[0] if pline else alt.Chart()
    for i in range(1, len(pline)):
        plines |= pline[i]
    
    # Configure chart once at the end
    plines = plines.configure_axis(grid=False)
    
    # Save the plot
    if save_path:
        plines.save(save_path)
        print(f"ROC curves saved to {save_path}")
    
    if separate:
        return p_separate
    else:
        return plines


def plot_scatter(data: Dict, show_score: bool = True, ncols: int = 3,
                 width: int = 400, height: int = 400,
                 save_path: Optional[str] = None, separate: bool = False) -> alt.Chart:
    """Plot scatter plots for regression task evaluation.
    
    This function creates scatter plots to compare predicted vs. experimental values
    for regression tasks, with optional R² score display.
    
    Args:
        data: Dictionary containing scatter plot data for each model
        show_score: Whether to show the R² score on the plot
        ncols: Number of columns to arrange the plots
        width: Width of each plot
        height: Height of each plot
        save_path: Path to save the plot. If None, plot will be shown interactively
        separate: Whether to return separate plots for each model
        
    Returns:
        Altair chart object (combined or separate plots based on separate parameter)
    """
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
        r2 = scatter_data.pop('r2', 0)  # Use pop with default value
        
        # More efficient DataFrame creation
        # Original: ddot = pd.DataFrame(scatter_data)
        ddot = pd.DataFrame(scatter_data)
        
        # Create scatter plot with optimized encoding
        dot = alt.Chart(ddot, title=model).mark_point(filled=True).encode(
            x=alt.X("predicted:Q"),
            y=alt.Y("experiment:Q"),
        ).properties(width=width, height=height)
        
        if show_score:
            # More efficient text positioning calculation
            # Original: min_x = ddot['predicted'].min(); max_y = ddot['experiment'].max()
            min_x = ddot['predicted'].min()
            max_y = ddot['experiment'].max()
            
            text = alt.Chart().mark_text(size=14, align="left", baseline="bottom").encode(
                x=alt.datum(min_x + 0.5),
                y=alt.datum(max_y - 0.5),
                text=alt.datum(f"R²={r2:.3f}")  # Format R² value
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
    pdots = pdot[0] if pdot else alt.Chart()
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


def plot_attention_map(attentions: Union[Tuple, List], sequences: List[str], tokenizer,
                       seq_idx: int = 0, layer: int = -1, head: int = -1,
                       width: int = 800, height: int = 800,
                       save_path: Optional[str] = None) -> alt.Chart:
    """Plot attention map visualization for transformer models.
    
    This function creates a heatmap visualization of attention weights between tokens
    in a sequence, showing how the model attends to different parts of the input.
    
    Args:
        attentions: Tuple or list containing attention weights from model layers
        sequences: List of input sequences
        tokenizer: Tokenizer object for converting tokens to readable text
        seq_idx: Index of the sequence to plot, default 0
        layer: Layer index to visualize, default -1 (last layer)
        attention_head: Attention head index to visualize, default -1 (last head)
        width: Width of the plot
        height: Height of the plot
        save_path: Path to save the plot. If None, plot will be shown interactively
        
    Returns:
        Altair chart object showing the attention heatmap
    """
    # More efficient attention data extraction with numpy
    # Original: attn_layer = attentions[layer].numpy()
    attn_layer = np.array(attentions[layer])
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
    df_data = {
        "token1": [f"{str(i).zfill(flen)}{t1}" for i, t1 in enumerate(tokens) for _ in range(num_tokens)],
        'token2': [f"{str(num_tokens-j).zfill(flen)}{t2}" for _ in range(num_tokens) for j, t2 in enumerate(tokens)],
        'attn': [attn_head[i][j] for i in range(num_tokens) for j in range(num_tokens)]
    }
    
    # More efficient DataFrame creation
    # Original: source = pd.DataFrame(df)
    source = pd.DataFrame(df_data)
    
    # Enable VegaFusion for Altair performance
    alt.data_transformers.enable("vegafusion")
    
    # Create attention map with optimized encoding and axis configuration
    # Original: Multiple axis configurations
    attn_map = alt.Chart(source).mark_rect().encode(
        x=alt.X('token1:O', axis=alt.Axis(
            labelExpr=f"substring(datum.value, {flen}, 100)",
            labelAngle=-45,
        )).title(None),
        y=alt.Y('token2:O', axis=alt.Axis(
            labelExpr=f"substring(datum.value, {flen}, 100)",
            labelAngle=0,
        )).title(None),
        color=alt.Color('attn:Q').scale(scheme='viridis'),
    ).properties(
        width=width,
        height=height
    ).configure_axis(grid=False)
    
    # Save the plot
    if save_path:
        attn_map.save(save_path)
        print(f"Attention map saved to {save_path}")
    
    return attn_map


def plot_embeddings(hidden_states: Union[Tuple, List], attention_mask: Union[Tuple, List], 
                    reducer: str = "t-SNE",
                    labels: Union[Tuple, List] = None, label_names: Union[str, List] = None,
                    ncols: int = 4, width: int = 300, height: int = 300,
                    save_path: Optional[str] = None, separate: bool = False) -> alt.Chart:
    """Visualize embeddings using dimensionality reduction techniques.
    
    This function creates 2D visualizations of high-dimensional embeddings from different
    model layers using PCA, t-SNE, or UMAP dimensionality reduction methods.
    
    Args:
        hidden_states: Tuple or list containing hidden states from model layers
        attention_mask: Tuple or list containing attention masks for sequence padding
        reducer: Dimensionality reduction method. Options: 'PCA', 't-SNE', 'UMAP'
        labels: List of labels for the data points
        labels_names: List of label names for legend display
        ncols: Number of columns to arrange the plots
        width: Width of each plot
        height: Height of each plot
        save_path: Path to save the plot. If None, plot will be shown interactively
        separate: Whether to return separate plots for each layer
        
    Returns:
        Altair chart object (combined or separate plots based on separate parameter)
        
    Raises:
        ValueError: If unsupported dimensionality reduction method is specified
    """
    import torch
    
    # More efficient dimensionality reducer selection with error handling
    # Original: Multiple if-elif statements
    reducer_map = {
        "pca": ("sklearn.decomposition", "PCA"),
        "t-sne": ("sklearn.manifold", "TSNE"),
        "umap": ("umap", "UMAP")
    }
    
    reducer_lower = reducer.lower()
    if reducer_lower not in reducer_map:
        raise ValueError(f"Unsupported dim reducer '{reducer}', please try PCA, t-SNE or UMAP.")
    
    module_name, class_name = reducer_map[reducer_lower]
    
    try:
        if module_name == "sklearn.decomposition":
            from sklearn.decomposition import PCA
            dim_reducer = PCA(n_components=2)
        elif module_name == "sklearn.manifold":
            from sklearn.manifold import TSNE
            dim_reducer = TSNE(n_components=2)
        elif module_name == "umap":
            from umap import UMAP
            dim_reducer = UMAP(n_components=2)
    except ImportError as e:
        raise ImportError(f"Required package for {reducer} not installed: {e}")
    
    # Pre-allocate plot dictionaries for better memory management
    # Original: pdot = {}; p_separate = {}
    pdot = {}
    p_separate = {}
    
    # More efficient embedding processing with numpy operations
    # Original: Multiple numpy conversions and calculations
    for i, (hidden, mask) in enumerate(zip(hidden_states, attention_mask)):
        # Convert to numpy once and use vectorized operations
        embeddings = np.array(hidden)
        attention_mask_array = np.array(mask)
        
        # More efficient mean calculation with numpy operations
        # Original: torch.sum(attention_mask*embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
        mask_sum = np.sum(attention_mask_array, axis=1, keepdims=True)
        mean_embeddings = np.sum(attention_mask_array[..., None] * embeddings, axis=-2) / mask_sum
        
        # More efficient dimensionality reduction
        layer_dim_reduced_vectors = dim_reducer.fit_transform(mean_embeddings)
        
        # Better label handling with default values
        if not labels:
            labels = ["Uncategorized"] * layer_dim_reduced_vectors.shape[0]
        
        # More efficient DataFrame creation with list comprehension
        # Original: Multiple dictionary assignments
        df_data = {
            'Dimension 1': layer_dim_reduced_vectors[:, 0],
            'Dimension 2': layer_dim_reduced_vectors[:, 1],
            'labels': [label_names[int(i)] if label_names else str(i) for i in labels]
        }
        
        source = pd.DataFrame(df_data)
        
        # Create plot with optimized encoding
        dot = alt.Chart(source, title=f"Layer {i+1}").mark_point(filled=True).encode(
            x=alt.X("Dimension 1:Q"),
            y=alt.Y("Dimension 2:Q"),
            color=alt.Color("labels:N", legend=alt.Legend(title="Labels")),
        ).properties(width=width, height=height)
        
        if separate:
            p_separate[f"Layer{i+1}"] = dot.configure_axis(grid=False)
        
        # More efficient plot arrangement
        idx = i // ncols
        if i % ncols == 0:
            pdot[idx] = dot
        else:
            pdot[idx] |= dot
    
    # More efficient plot combination
    # Original: Multiple conditional checks and assignments
    pdots = pdot[0] if pdot else alt.Chart()
    for i in range(1, len(pdot)):
        pdots &= pdot[i]
    
    # Configure chart once at the end
    pdots = pdots.configure_axis(grid=False)
    
    # Save the plot
    if save_path:
        pdots.save(save_path)
        print(f"Embeddings visualization saved to {save_path}")
    
    if separate:
        return p_separate
    else:
        return pdots


def plot_muts(data: Dict, show_score: bool = False,
              width: Optional[int] = None, height: int = 100,
              save_path: Optional[str] = None) -> alt.Chart:
    """Visualize mutation effects on model predictions.
    
    This function creates comprehensive visualizations of how different mutations
    affect model predictions, including:
    - Heatmap showing mutation effects at each position
    - Line plot showing gain/loss of function
    - Bar chart showing maximum effect mutations
    
    Args:
        data: Dictionary containing mutation data with 'raw' and mutation keys
        show_score: Whether to show the score values on the plot (currently not implemented)
        width: Width of the plot. If None, automatically calculated based on sequence length
        height: Height of the plot
        save_path: Path to save the plot. If None, plot will be shown interactively
        
    Returns:
        Altair chart object showing the combined mutation effects visualization
    """
    # More efficient data extraction and preprocessing
    # Original: seqlen = len(data['raw']['sequence']); flen = len(str(seqlen))
    raw_data = data['raw']
    sequence = raw_data['sequence']
    seqlen = len(sequence)
    flen = len(str(seqlen))
    
    # Use list comprehension for more efficient data creation
    # Original: mut_list = [x for x in data.keys() if x != 'raw']; raw_bases = [base for base in data['raw']['sequence']]
    mut_list = [x for x in data.keys() if x != 'raw']
    raw_bases = list(sequence)
    
    # Pre-allocate data structures for better performance
    # Original: Multiple dictionary initializations with empty lists
    dheat = defaultdict(list)
    dline = {
        "x": [f"{str(i).zfill(flen)}{x}" for i, x in enumerate(raw_bases)] * 2,
        "score": [0.0] * seqlen * 2,
        "type": ["gain"] * seqlen + ["loss"] * seqlen
    }
    dbar = defaultdict(list)
    
    # More efficient mutation processing with optimized loops
    # Original: Multiple nested loops with repeated condition checks
    for i, base1 in enumerate(raw_bases):
        # Pre-calculate mutation prefixes for better performance
        ref = f"mut_{i}_{base1}_{base1}"
        mut_prefix = f"mut_{i}_{base1}_"
        
        # More efficient score tracking with numpy operations
        maxabs = 0.0
        maxscore = 0.0
        maxabs_index = base1
        
        # Filter mutations once and process efficiently
        relevant_muts = [x for x in mut_list if x.startswith(mut_prefix)] + [ref]
        
        for mut in sorted(relevant_muts):
            if mut in data:
                base2 = mut.split("_")[-1]
                score = data[mut]['score']
                
                # Batch data collection for better performance
                dheat["base"].append(f"{str(i).zfill(flen)}{base1}")
                dheat["mut"].append(base2)
                dheat["score"].append(score)
                
                # More efficient score accumulation
                if score >= 0:
                    dline["score"][i] += score
                else:
                    dline["score"][i + seqlen] -= score
                
                # Track maximum absolute score more efficiently
                if abs(score) > maxabs:
                    maxabs = abs(score)
                    maxscore = score
                    maxabs_index = base2
            else:
                # Handle missing mutations more efficiently
                dheat["base"].append(f"{str(i).zfill(flen)}{base1}")
                dheat["mut"].append(base1)
                dheat["score"].append(0.0)
        
        # Collect bar chart data efficiently
        dbar["x"].append(f"{str(i).zfill(flen)}{base1}")
        dbar["score"].append(maxscore)
        dbar["base"].append(maxabs_index)
        
        # Process deletion and insertion mutations more efficiently
        del_prefix = f"del_{i}_"
        ins_prefix = f"ins_{i}_"
        
        for mut in [x for x in mut_list if x.startswith(del_prefix)]:
            base2 = f"del_{mut.split('_')[-1]}"
            score = data[mut]['score']
            dheat["base"].append(f"{str(i).zfill(flen)}{base1}")
            dheat["mut"].append(base2)
            dheat["score"].append(score)
        
        for mut in [x for x in mut_list if x.startswith(ins_prefix)]:
            base2 = f"ins_{mut.split('_')[-1]}"
            score = data[mut]['score']
            dheat["base"].append(f"{str(i).zfill(flen)}{base1}")
            dheat["mut"].append(base2)
            dheat["score"].append(score)
    
    # More efficient color domain calculation with numpy
    # Original: Multiple min/max calculations and list comprehensions
    if dheat['score']:
        all_scores = dheat['score']
        max_abs_score = max(abs(min(all_scores)), abs(max(all_scores)))
        domain1 = [-max_abs_score, 0.0, max_abs_score]
        range1_ = ['#2166ac', '#f7f7f7', '#b2182b']
        
        # More efficient unique value extraction
        unique_bases = sorted(set(dbar['base']))
        range2_ = ["#33a02c", "#e31a1c", "#1f78b4", "#ff7f00", "#cab2d6"][:len(unique_bases)]
        
        # Enable VegaFusion for Altair performance
        alt.data_transformers.enable("vegafusion")
        
        # Calculate width automatically if not provided
        if width is None:
            width = int(height * len(raw_bases) / len(set(dheat['mut'])))
        
        # Create heatmap with optimized encoding
        if dheat['base']:
            pheat = alt.Chart(pd.DataFrame(dict(dheat))).mark_rect().encode(
                x=alt.X('base:O', axis=alt.Axis(
                    labelExpr=f"substring(datum.value, {flen}, {flen}+1)",
                    labelAngle=0,
                )).title(None),
                y=alt.Y('mut:O').title("mutation"),
                color=alt.Color('score:Q').scale(domain=domain1, range=range1_),
            ).properties(width=width, height=height)
            
            # Create line plot with optimized encoding
            pline = alt.Chart(pd.DataFrame(dline)).mark_line().encode(
                x=alt.X('x:O').title(None).axis(labels=False),
                y=alt.Y('score:Q'),
                color=alt.Color('type:N').scale(
                    domain=['gain', 'loss'], range=['#b2182b', '#2166ac']
                ),
            ).properties(width=width, height=height)
            
            # Create bar chart with optimized encoding
            pbar = alt.Chart(pd.DataFrame(dict(dbar))).mark_bar().encode(
                x=alt.X('x:O').title(None).axis(labels=False),
                y=alt.Y('score:Q'),
                color=alt.Color('base:N').scale(
                    domain=unique_bases, range=range2_
                ),
            ).properties(width=width, height=height)
            
            # Combine plots efficiently
            pmerge = pheat & pbar & pline
            pmerge = pmerge.configure_axis(grid=False)
            
            # Save the plot
            if save_path:
                pmerge.save(save_path)
                print(f"Mutation effects visualization saved to {save_path}")
            
            return pmerge
    
    # Return empty chart if no data
    return alt.Chart()

