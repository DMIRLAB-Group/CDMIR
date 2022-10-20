from enum import Enum


class Mark(Enum):
    Tail = -1
    NULL = 0
    ARROW = 1
    CIRCLE = 2
    def __str__(self):
        return self.name

    @staticmethod
    def pdag_marks():
        return [Mark.Tail, Mark.ARROW]

    @staticmethod
    def pag_marks():
        return [Mark.Tail, Mark.ARROW, Mark.CIRCLE]