import matplotlib.pyplot as plt
from matplotlib import patches


def plot_graph(graph, layout, is_latent=None, figsize=None, dpi=None):
    if is_latent is None:
        is_latent = set()

    node_radius = 0.05

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)
    ax.set_axis_off()
    center = (0.5, 0.5)
    pos = layout(graph)
    for node in graph.nodes:
        if node in is_latent:
            node_shape = 'rectangle'
        else:
            node_shape = 'circle'

        node_str = str(node)
        plt_node(ax, (pos[node_str][0] + center[0], pos[node_str][1] + center[1]), node_radius=node_radius,
                 node_shape=node_shape, node_name=node_str)

    plt.show()
    return fig


def plt_node(axes,
             pos,
             node_radius,
             node_shape,
             node_name='',
             line_color='black',
             fill_color='white',
             text_color='black'):
    print(pos)

    if node_shape == 'rectangle':
        axes.add_patch(patches.Rectangle((pos[0] - node_radius, pos[1] - node_radius), node_radius * 2, node_radius * 2,
                                         facecolor=fill_color, edgecolor=line_color))
    elif node_shape == 'circle':
        axes.add_patch(patches.Circle(pos, node_radius, facecolor=fill_color, edgecolor=line_color))
    else:
        raise NotImplementedError(f'Node Shape {node_shape} is not implemented.')
