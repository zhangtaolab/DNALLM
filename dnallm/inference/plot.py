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


def plot_bars(data: dict, show_score: bool = True, ncols: int = 3,
              width: int = 200, height: int = 50, bar_width: int = 30,
              domain: Union[tuple, list]= (0.0, 1.0),
              save_path: str = None) -> alt.Chart:
    """
    Plot bar chart.
    Args:
        data (dict): Data to be plotted.
        show_score (bool): Whether to show the score on the plot.
        ncols (int): Number of columns in the plot.
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
        idx = n // ncols
        if n % ncols == 0:
            pbar[idx] = p
        else:
            pbar[idx] |= p
    # Combine the plots
    for i, p in enumerate(pbar):
        if i == 0:
            pbars = pbar[p]
        else:
            pbars &= pbar[p]
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


def plot_scatter(data: dict, show_score: bool = True, ncols: int = 3,
                 width: int = 400, height: int = 400,
                 save_path: str = None) -> alt.Chart:
    """
    Plot scatter chart.
    Args:
        data (dict): Data to be plotted.
        show_score (bool): Whether to show the score on the plot.
        ncols (int): Number of columns in the plot.
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
        idx = n // ncols
        if n % ncols == 0:
            pdot[idx] = p
        else:
            pdot[idx] |= p
    # Combine the plots
    for i, p in enumerate(pdot):
        if i == 0:
            pdots = pdot[p]
        else:
            pdots &= pdot[p]
    # Configure the chart
    pdots = pdots.configure_axis(grid=False)
    # Save the plot
    if save_path:
        pdots.save(save_path)
        print(f"Metrics scatter plots saved to {save_path}")
    return pdots


def plot_attention_map(attentions: Union[tuple, list], sequences: list, tokenizer,
                       seq_idx: int=0, layer: int=-1, head: int=-1,
                       width: int = 800, height: int = 800,
                       save_path: str = None) -> alt.Chart:
    """
    Plot attention map.
    Args:
        attentions (tuple): Tuple containing attention weights.
        sequences (list): List of sequences.
        tokenizer: Tokenizer object.
        seq_idx (int): Index of the sequence to plot.
        layer (int): Layer index.
        head (int): Head index.
        width (int): Width of the plot.
        height (int): Height of the plot.
        save_path (str): Path to save the plot.
    Returns:
        attn_map: Altair chart object.
    """
    # Plot attention map
    attn_layer = attentions[layer].numpy()
    attn_head = attn_layer[seq_idx][head]
    # Get the tokens
    seq = sequences[seq_idx]
    tokens_id = tokenizer.encode(seq)
    try:
        tokens = tokenizer.convert_ids_to_tokens(tokens_id)
    except:
        tokens = tokenizer.decode(seq).split()
    # Create a DataFrame for the attention map
    df = {"token1": [], 'token2': [], 'attn': []}
    for i, t1 in enumerate(tokens):
        for j, t2 in enumerate(tokens):
            df["token1"].append(t1)
            df["token2"].append(t2)
            df["attn"].append(attn_head[i][j])
    source = pd.DataFrame(df)
    # Enable VegaFusion for Altair
    alt.data_transformers.enable("vegafusion")
    # Plot the attention map
    attn_map = alt.Chart(source).mark_rect().encode(
        x=alt.X('token1:O').title(None),
        y=alt.X('token2:O').title(None),
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


def plot_embeddings(hidden_states: Union[tuple, list], attention_mask: Union[tuple, list], reducer: str="t-SNE",
                    labels: Union[tuple, list]=None, label_names: Union[str, list]=None,
                    ncols: int=4, width: int=300, height: int=300,
                    save_path: str = None) -> alt.Chart:
    '''
    Plot 
    '''
    import torch
    if reducer.lower() == "pca":
        from sklearn.decomposition import PCA
        dim_reducer = PCA(n_components=2)
    elif reducer.lower() == "t-sne":
        from sklearn.manifold import TSNE
        dim_reducer = TSNE(n_components=2)
    elif reducer.lower() == "umap":
        from umap import UMAP
        dim_reducer = UMAP(n_components=2)
    else:
        raise("Unsupported dim reducer, please try PCA, t-SNE or UMAP.")
    
    pdot = {}
    for i, hidden in enumerate(hidden_states):
        embeddings = hidden.numpy()
        mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
        layer_dim_reduced_vectors = dim_reducer.fit_transform(mean_sequence_embeddings.numpy())
        if len(labels) == 0:
            labels = ["Uncategorized"] * layer_dim_reduced_vectors.shape[0]
        df = {
            'Dimension 1': layer_dim_reduced_vectors[:,0],
            'Dimension 2': layer_dim_reduced_vectors[:,1],
            'labels': [label_names[int(i)] for i in labels]
        }
        source = pd.DataFrame(df)
        dot = alt.Chart(source, title=f"Layer {i+1}").mark_point(filled=True).encode(
            x=alt.X("Dimension 1:Q"),
            y=alt.Y("Dimension 2:Q"),
            color=alt.Color("labels:N", legend=alt.Legend(title="Labels")),
        ).properties(width=width, height=height)
        idx = i // ncols
        if i % ncols == 0:
            pdot[idx] = dot
        else:
            pdot[idx] |= dot
    # Combine the plots
    for i, p in enumerate(pdot):
        if i == 0:
            pdots = pdot[p]
        else:
            pdots &= pdot[p]
    # Configure the chart
    pdots = pdots.configure_axis(grid=False)
    # Save the plot
    if save_path:
        pdots.save(save_path)
        print(f"Embeddings visualization saved to {save_path}")
    return pdots
