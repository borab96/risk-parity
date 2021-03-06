import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
from rpp.utils import cluster_corr
import os


def combine_outs(figs):
    filename = f"{os.path.join(os.getcwd(), 'plots/combined.html')}"
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    include_plotlyjs = True
    for fig in figs:
        # fig.show('browser')
        inner_html = fig.to_html(include_plotlyjs=include_plotlyjs).split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
        include_plotlyjs = False
        dashboard.write("</body></html>" + "\n")
    return filename


def tree_plot(df, width=800, height=400, **kwargs):
    fig = ff.create_dendrogram(df, labels=df.columns, **kwargs)
    fig.update_layout(width=width, height=height)
    return fig


def heat_map(*args, cluster=True, **kwargs):
    if cluster:
        # data_dist = pdist(args[0], 'correlation')
        return px.imshow(cluster_corr(args[0]), *args[1:], **kwargs)
    else:
        return px.imshow(*args, **kwargs)


def full_heat_map(df):
    """
    From https://stackoverflow.com/questions/66547583/plotly-clustered-heatmap-with-dendrogram-python.
    Produces clustered heatmap from covaraince matrix.

    :param df: covariance dataframe
    :param dark_theme: If true, darkens background
    :return: plotly figure
    """
    fig = ff.create_dendrogram(df.values, orientation='bottom')
    fig.for_each_trace(lambda trace: trace.update(visible=False))
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'
    dendro_side = ff.create_dendrogram(df.values, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'
    # Add Side Dendrogram Data to Figure
    for data in dendro_side['data']:
        fig.add_trace(data)
    # Create Heatmap
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))
    data_dist = pdist(df.values)
    heat_data = squareform(data_dist)
    heat_data = heat_data[dendro_leaves, :]
    heat_data = heat_data[:, dendro_leaves]
    heatmap = [
        go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data
            # colorscale='Reds'
        )
    ]
    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']
    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)
    # Edit Layout
    fig.update_layout({'width': 800, 'height': 800,
                       'showlegend': False, 'hovermode': 'closest',
                       })
    # Edit xaxis
    fig.update_layout(xaxis={'domain': [.15, 1],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'ticks': ""})
    # Edit xaxis2
    fig.update_layout(xaxis2={'domain': [0, .15],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""})

    # Edit yaxis
    fig.update_layout(yaxis={'domain': [0, 1],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'showticklabels': False,
                             'ticks': ""
                             })
    # # Edit yaxis2
    fig.update_layout(yaxis2={'domain': [.825, .975],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_tickfont=dict(color='rgba(0,0,0,0)'))
    return fig


def plot(*args, **kwargs):
    return px.line(*args, width=1400, height=800, **kwargs)


def bar(*args, **kwargs):
    fig = px.bar(*args, width=1400, height=800, **kwargs)
    fig.update_xaxes(rangeslider_visible=True)
    return fig
