from typing import List


class Interval(object):
    def __init__(self, lower: float, upper: float):
        if lower > upper:
            raise ValueError('lower has to be smaller or equal to upper')

        self.lower = lower
        self.upper = upper

    def __eq__(self, other):
        return self.lower == self.upper and self.upper == other.upper

    def __add__(self, other):
        return Interval(
            self.lower + other.lower,
            self.upper + other.upper
        )

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        options = {self.lower * other.lower,
                   self.lower * other.upper,
                   self.upper * other.lower,
                   self.upper * other.upper}
        return Interval(
            min(options),
            max(options)
        )

    def __pos__(self):
        return Interval(
            self.lower,
            self.upper
        )

    def __neg__(self):
        return Interval(
            -self.upper,
            -self.lower
        )

    def __repr__(self):
        return f'Interval({self.lower},{self.upper})'


class IntervalMatrix(object):
    def __init__(self, interval_matrix: List[List[Interval]]):
        if len(interval_matrix) == 0:
            raise ValueError('interval_matrix cannot be empty')

        if not all(len(row) == len(interval_matrix[0]) for row in interval_matrix):
            raise ValueError('all rows in interval_matrix need to have the same length')

        self.matrix = interval_matrix

    def __add__(self, other):
        if self.shape() != other.shape:
            raise ValueError(f'shape mismatch between {self.shape()} and {other.shape()}')
        return IntervalMatrix(
            [[col1 + col2 for col1, col2 in zip(row1, row2)]
             for row1, row2 in zip(self.matrix, other.matrix)]
        )

    def shape(self):
        return (len(self.matrix), len(self.matrix[0]))
