# -*- coding: utf-8 -*-

class Matrix():
    def __init__(self, data):
        if type(data) is not list:
            raise TypeError('Matrix object requires list of data')
        elif type(data[0]) is not list:
            raise TypeError('Row of matrix should be list of value')

        for row_data in data:
            if len(row_data) != len(data[0]):
                raise ValueError('All num of cols should be the same')

        # 全てfloat型にする
        self.data = \
            [
                [
                    float(value)
                    for value in row_data
                ]
                for row_data in data
            ]

    @property
    def row(self):
        return len(self.data)

    @property
    def col(self):
        return len(self.data[0])

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        ret_str = '['
        for ir, row_data in enumerate(self.data):
            if ir == 0:
                ret_str += '[ '
            else:
                ret_str += ' [ '

            for cols_value in row_data:
                ret_str += str(cols_value) + ' '

            if ir != len(self.data) - 1:
                ret_str += ']\n'
            else:
                ret_str += ']'
        ret_str += ']'
        return ret_str

    def __add__(self, other):
        u"""全要素をそれぞれ加算する"""
        self._verificate_shape_is_same(other)
        return Matrix(self._calc_all_elements(
            other, lambda l, r: l + r))

    def __sub__(self, other):
        u"""全要素をそれぞれ減算する"""
        self._verificate_shape_is_same(other)
        return Matrix(self._calc_all_elements(
            other, lambda l, r: l - r))

    def __mul__(self, other):
        u"""全要素をそれぞれ乗算する"""
        self._verificate_shape_is_same(other)
        return Matrix(self._calc_all_elements(
            other, lambda l, r: l * r))

    def __truediv__(self, other):
        u"""全要素をそれぞれ乗算する"""
        self._verificate_shape_is_same(other)
        return Matrix(self._calc_all_elements(
            other, lambda l, r: l / r))

    def _verificate_shape_is_same(self, other):
        if self.col != other.col or self.row != other.row:
            raise ValueError('Shape of both should be same')

    def _calc_all_elements(self, other, operator):
        return \
            [
                [
                    operator(self.data[ir][ic], other.data[ir][ic])
                    for ic in range(self.col)
                ]
                for ir in range(self.row)
            ]
