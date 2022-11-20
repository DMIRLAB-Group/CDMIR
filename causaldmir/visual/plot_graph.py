import matplotlib.pyplot as plt
from matplotlib import patches
from numpy import abs, cos, sin, sqrt, sign

from causaldmir.graph import Mark


def plot_graph(graph, layout, is_latent=None, figsize=None, dpi=300, node_radius=0.04, edge_circle_mark_ratio=1 / 6):
    if is_latent is None:
        is_latent = set()

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
    ax.set_axis_off()
    center = (0.5, 0.5)
    pos = layout(graph)
    node_str = [str(node) for node in graph.nodes]
    for i, node in enumerate(graph.nodes):
        if node in is_latent:
            node_fill_color = 'white'
        else:
            node_fill_color = '0.8'  # light gray

        plt_node(ax, (pos[node][0] + center[0], pos[node][1] + center[1]), node_radius=node_radius,
                 node_name=node_str[i], node_fill_color=node_fill_color)

    edge_circle_mark_radius = node_radius * edge_circle_mark_ratio
    for edge in graph.edges:
        node_u, node_v, mark_u, mark_v = edge
        plt_edge(ax,
                 (pos[node_u][0] + center[0], pos[node_u][1] + center[1]),
                 (pos[node_v][0] + center[0], pos[node_v][1] + center[1]),
                 mark_u, mark_v,
                 node_radius=node_radius,
                 circle_mark_radius=edge_circle_mark_radius)

    return fig


def plt_edge(axes,
             pos_u,
             pos_v,
             mark_u,
             mark_v,
             node_radius,
             circle_mark_radius,
             ):
    assert circle_mark_radius > 0

    dx = pos_v[0] - pos_u[0]
    dy = pos_v[1] - pos_u[1]
    dis = sqrt(dx ** 2 + dy ** 2)
    offset_x = node_radius * dx / dis
    offset_y = node_radius * dy / dis

    pos_u = (pos_u[0] + offset_x, pos_u[1] + offset_y)
    pos_v = (pos_v[0] - offset_x, pos_v[1] - offset_y)

    offset_x = 2 * circle_mark_radius * dx / dis
    offset_y = 2 * circle_mark_radius * dy / dis

    if mark_u == Mark.Circle:
        axes.add_patch(
            patches.Circle((pos_u[0] + offset_x / 2, pos_u[1] + offset_y / 2), circle_mark_radius, facecolor='white',
                           edgecolor='black'))
        pos_u = (pos_u[0] + offset_x, pos_u[1] + offset_y)
    if mark_v == Mark.Circle:
        axes.add_patch(
            patches.Circle((pos_v[0] - offset_x / 2, pos_v[1] - offset_y / 2), circle_mark_radius, facecolor='white',
                           edgecolor='black'))
        pos_v = (pos_v[0] - offset_x, pos_v[1] - offset_y)

    if mark_u == Mark.Arrow:
        if mark_v == Mark.Arrow:
            arrow_style = '<|-|>'
        else:
            arrow_style = '<|-'
    elif mark_v == Mark.Arrow:
        arrow_style = '-|>'
    else:
        arrow_style = '-'
    axes.add_patch(patches.FancyArrowPatch(pos_u, pos_v,
                                           edgecolor='black', facecolor='black',
                                           arrowstyle=arrow_style, mutation_scale=12, shrinkA=0, shrinkB=0))


def plt_node(axes,
             pos,
             node_radius,
             node_name='',
             edge_color='black',
             node_fill_color='white',
             font_family='sans-serif',
             font_size=8,
             font_color='black',
             ):
    axes.add_patch(patches.Circle(pos, node_radius, facecolor=node_fill_color, edgecolor=edge_color))

    axes.text(pos[0], pos[1], node_name, ha='center', va='center_baseline', family=font_family, size=font_size,
              color=font_color)
