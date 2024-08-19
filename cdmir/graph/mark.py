from enum import IntEnum


class Mark(IntEnum):
    Tail = -1
    Null = 0
    Arrow = 1
    Circle = 2

    def __str__(self):
        return self.name

    @staticmethod
    def pdag_marks():
        return [Mark.Tail, Mark.Arrow]

    @staticmethod
    def pag_marks():
        return [Mark.Tail, Mark.Arrow, Mark.Circle]
