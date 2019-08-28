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
