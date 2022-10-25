from collections import namedtuple

from .mark import Mark


class Edge(namedtuple('Edge', ['node_u', 'node_v', 'mark_u', 'mark_v'])):
    __slots__ = ()
    def __new__(cls, node_u, node_v, mark_u=Mark.Tail, mark_v=Mark.ARROW):
        return super().__new__(cls, node_u, node_v, mark_u, mark_v)
    def __str__(self):
        return f'{self.node_u} {_lmark2ascii[self.mark_u]}-{_rmark2ascii[self.mark_v]} {self.node_v}'


_lmark2ascii = {
    Mark.Tail: '-',
    Mark.ARROW: '<',
    Mark.CIRCLE: 'o'
}
_rmark2ascii = {
    Mark.Tail: '-',
    Mark.ARROW: '>',
    Mark.CIRCLE: 'o'
}
