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
    if task_type in ['binary', 'multiclass', 'multilabel', 'token']:
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
              save_path: str = None, separate: bool=False) -> alt.Chart:
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
    p_separate = {}
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
        if separate:
            p_separate[metric] = p.configure_axis(grid=False)
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
    if separate:
        return p_separate
    else:
        return pbars


def plot_curve(data: dict, show_score: bool = True,
               width: int = 400, height: int = 400,
               save_path: str = None, separate: bool=False) -> alt.Chart:
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
    p_separate = {}
    # Plot ROC curve
    roc_data = pd.DataFrame(data['ROC'])
    pline[0] = alt.Chart(roc_data).mark_line().encode(
        x=alt.X("fpr").scale(domain=(0.0, 1.0)),
        y=alt.Y("tpr").scale(domain=(0.0, 1.0)),
        color="models",
    ).properties(width=width, height=height)
    if separate:
        p_separate['ROC'] = pline[0]
    # Plot PR curve
    pr_data = pd.DataFrame(data['PR'])
    pline[1] = alt.Chart(pr_data).mark_line().encode(
        x=alt.X("recall").scale(domain=(0.0, 1.0)),
        y=alt.Y("precision").scale(domain=(0.0, 1.0)),
        color="models",
    ).properties(width=width, height=height)
    if separate:
        p_separate['PR'] = pline[1]
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
    if separate:
        return p_separate
    else:
        return plines


def plot_scatter(data: dict, show_score: bool = True, ncols: int = 3,
                 width: int = 400, height: int = 400,
                 save_path: str = None, separate: bool=False) -> alt.Chart:
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
    p_separate = {}
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
        if separate:
            p_separate[model] = p.configure_axis(grid=False)
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
    if separate:
        return p_separate
    else:
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
    num_tokens = len(tokens)
    flen = len(str(num_tokens))
    df = {"token1": [], 'token2': [], 'attn': []}
    for i, t1 in enumerate(tokens):
        for j, t2 in enumerate(tokens):
            df["token1"].append(str(i).zfill(flen)+t1)
            df["token2"].append(str(num_tokens-j).zfill(flen)+t2)
            df["attn"].append(attn_head[i][j])
    source = pd.DataFrame(df)
    # Enable VegaFusion for Altair
    alt.data_transformers.enable("vegafusion")
    # Plot the attention map
    attn_map = alt.Chart(source).mark_rect().encode(
        x=alt.X('token1:O', axis=alt.Axis(
                    labelExpr = f"substring(datum.value, {flen}, 100)",
                    labelAngle=-45,
                    )
                ).title(None),
        y=alt.Y('token2:O', axis=alt.Axis(
                    labelExpr = f"substring(datum.value, {flen}, 100)",
                    labelAngle=0,
                    )
                ).title(None),
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
                    save_path: str = None, separate: bool=False) -> alt.Chart:
    '''
    Visualize embeddings
    
    Args:
        hidden_states (tuple): Tuple containing hidden states.
        attention_mask (tuple): Tuple containing attention mask.
        reducer (str): Dimensionality reduction method. Options: PCA, t-SNE, UMAP.
        labels (list): List of labels for the data points.
        label_names (list): List of label names.
        ncols (int): Number of columns in the plot.
        width (int): Width of the plot.
        height (int): Height of the plot.
        save_path (str): Path to save the plot.
        separate (bool): Whether to return separate plots for each layer.
    Returns:
        pdots: Altair chart object.
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
    p_separate = {}
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
        if separate:
            p_separate[f"Layer{i+1}"] = dot.configure_axis(grid=False)
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
    if separate:
        return p_separate
    else:
        return pdots


def plot_muts(data: dict, show_score: bool = False,
              width: int = None, height: int = 100,
              save_path: str = None) -> alt.Chart:
    '''
    Visualize mutation effects
    '''
    # Create dataframe
    seqlen = len(data['raw']['sequence'])
    flen = len(str(seqlen))
    mut_list = [x for x in data.keys() if x != 'raw']
    raw_bases = [base for base in data['raw']['sequence']]
    dheat = {"base": [], 'mut': [], 'score': []}
    dline = {"x": [str(i).zfill(flen)+x for i,x in enumerate(raw_bases)] * 2,
             "score": [0.0]*seqlen*2,
             "type": ["gain"]*seqlen + ["loss"]*seqlen}
    dbar = {"x": [], "score": [], "base": []}
    # Iterate through mutations
    for i, base1 in enumerate(raw_bases):
        ref = "mut_" + str(i) + "_" + base1 + "_" + base1
        # replacement
        mut_prefix = "mut_" + str(i) + "_" + base1 + "_"
        maxabs = 0.0
        maxscore = 0.0
        maxabs_index = base1
        for mut in sorted([x for x in mut_list if x.startswith(mut_prefix)] + [ref]):
            if mut in data:
                # for heatmap
                base2 = mut.split("_")[-1]
                score = data[mut]['score']
                dheat["base"].append(str(i).zfill(flen)+base1)
                dheat["mut"].append(base2)
                dheat["score"].append(score)
                # for line
                if score >= 0:
                    dline["score"][i] += score
                elif score < 0:
                    dline["score"][i+seqlen] -= score
                # for bar chart
                if abs(score) > maxabs:
                    maxabs = abs(score)
                    maxscore = score
                    maxabs_index = base2
            else:
                dheat["base"].append(str(i).zfill(flen)+base1)
                dheat["mut"].append(base1)
                dheat["score"].append(0.0)
        # for bar chart
        dbar["x"].append(str(i).zfill(flen)+base1)
        dbar["score"].append(maxscore)
        dbar["base"].append(maxabs_index)
        # deletion
        del_prefix = "del_" + str(i) + "_"
        for mut in [x for x in mut_list if x.startswith(del_prefix)]:
            base2 = "del_" + mut.split("_")[-1]
            score = data[mut]['score']
            dheat["base"].append(str(i).zfill(flen)+base1)
            dheat["mut"].append(base2)
            dheat["score"].append(score)
        # insertion
        ins_prefix = "ins_" + str(i) + "_"
        for mut in [x for x in mut_list if x.startswith(ins_prefix)]:
            base2 = "ins_" + mut.split("_")[-1]
            score = data[mut]['score']
            dheat["base"].append(str(i).zfill(flen)+base1)
            dheat["mut"].append(base2)
            dheat["score"].append(score)
    # Set color domain and range
    domain1_min = min([data[mut]['score'] for mut in data])
    domain1_max = max([data[mut]['score'] for mut in data])
    domain1 = [-max([abs(domain1_min), abs(domain1_max)]),
               0.0,
               max([abs(domain1_min), abs(domain1_max)])]
    range1_ = ['#2166ac', '#f7f7f7', '#b2182b']
    domain2 = sorted([x for x in set(dbar['base'])])
    range2_ = ["#33a02c", "#e31a1c", "#1f78b4", "#ff7f00", "#cab2d6"][:len(domain2)]
    # Enable VegaFusion for Altair
    alt.data_transformers.enable("vegafusion")
    # Plot the heatmap
    if width is None:
        width = int(height * len(raw_bases) / len(set(dheat['mut'])))
    if dheat['base']:
        pheat = alt.Chart(pd.DataFrame(dheat)).mark_rect().encode(
            x=alt.X('base:O', axis=alt.Axis(
                    labelExpr = f"substring(datum.value, {flen}, {flen}+1)",
                    labelAngle=0,
                    )
                ).title(None),
            y=alt.Y('mut:O').title("mutation"),
            color=alt.Color('score:Q').scale(domain=domain1, range=range1_),
        ).properties(
            width=width, height=height
        )
        # Plot gain and loss
        pline = alt.Chart(pd.DataFrame(dline)).mark_line().encode(
            x=alt.X('x:O').title(None).axis(labels=False),
            y=alt.Y('score:Q'),
            color=alt.Color('type:N').scale(
                domain=['gain', 'loss'], range=['#b2182b', '#2166ac']
            ),
        ).properties(
            width=width, height=height
        )
        pbar = alt.Chart(pd.DataFrame(dbar)).mark_bar().encode(
            x=alt.X('x:O').title(None).axis(labels=False),
            y=alt.Y('score:Q'),
            color=alt.Color('base:N').scale(
                domain=domain2, range=range2_
            ),
        ).properties(
            width=width, height=height
        )
        pmerge = pheat & pbar & pline
        pmerge = pmerge.configure_axis(grid=False)
        # Save the plot
        if save_path:
            pmerge.save(save_path)
            print(f"Mutation effects visualization saved to {save_path}")
    return pheat

