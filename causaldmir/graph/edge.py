from collections import namedtuple

from .mark import Mark

Edge = namedtuple('Edge', ['node_u', 'node_v', 'mark_u', 'mark_v'])

_lmark2ascii = {
    Mark.Tail: '-',
    Mark.ARROW: '<',
    Mark.CIRCLE: 'o'
}
_rmark2ascii = {
    Mark.Tail: '-',
    Mark.ARROW: '>',
    Mark.CIRCLE: 'o'}


def edge2str(edge: Edge):
    return f'{edge.node_u} {_lmark2ascii[edge.mark_u]}-{_rmark2ascii[edge.mark_v]} {edge.node_v}'
