# -*- coding: utf-8 -*-


class Matrix():
    def __init__(self, values):
        if type(values) is not list:
            raise TypeError('Matrix object requires list of values')
        elif type(values[0]) is not list:
            raise TypeError('Row of matrix should be list of value')

        self.col = len(values[0])
        for row_values in values:
            if len(row_values) != self.col:
                raise ValueError('All num of cols should be the same')

        # 全てfloat型にする
        self._values = \
            [
                [
                    float(value) for value in row_values
                ]
                for row_values in values
            ]

        self.row = len(values)

    def __repr__(self):
        return 'Matrix()'

    def __str__(self):
        ret_str = '['
        for ir, row_values in enumerate(self._values):
            if ir == 0:
                ret_str += '[ '
            else:
                ret_str += ' [ '

            for cols_value in row_values:
                ret_str += str(cols_value) + ' '

            if ir != len(self._values) - 1:
                ret_str += ']\n'
            else:
                ret_str += ']'
        ret_str += ']'
        return ret_str

