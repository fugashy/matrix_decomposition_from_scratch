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

    def dot(self, other):
        u"""InnerProductを計算する"""
        # 同じ型同士でなければならない
        if not isinstance(other, Matrix):
            raise TypeError('dot function requires object type of Matrix')

        # 左の列と右の行が一致しているか
        if self.col != other.row:
            raise ValueError('dot requires the condiction self.col == self.row')

        return \
            [
                [
                    sum(
                        [
                            self.data[i][k] * other.data[k][j]
                            for k in range(self.col)
                        ])
                    for j in range(other.col)
                ]
                for i in range(self.row)
            ]

    def det(self):
        u"""行列式を計算する

        正方行列にのみ定義された量である

        3x3まではサラスの公式を使う
        それよりも大きい行列の場合は展開する
        """
        if self.col != self.row:
            raise ValueError('Col and Row should be the same to compute determinant')

        if self.row == 1:
            return self[0, 0]
        elif self.row == 2:
            return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        elif self.row == 3:
            return \
                (self[0, 0] * self[1, 1] * self[2, 2]) + \
                (self[0, 1] * self[1, 2] * self[2, 0]) + \
                (self[0, 2] * self[1, 0] * self[2, 1]) - \
                (self[0, 0] * self[1, 2] * self[2, 1]) - \
                (self[0, 1] * self[1, 0] * self[2, 2]) - \
                (self[0, 2] * self[1, 1] * self[2, 0])
        else:
            # 再帰的に展開してサラスの公式が使えるところまで落とす
            sum = 0.0
            for i in range(self.row):
                m = Matrix(
                    [
                        [
                            self[k, j]
                            for j in range(self.row)
                            if j != i
                        ]
                        for k in range(1, self.row)
                    ])
                sum += (-1.)**i * self[0, i] * m.det()
            return sum

    def sub(self, row, col):
        u"""部分行列を作る"""
        return Matrix(
            [
                [
                    self[i, j]
                    for j in range(self.col)
                    if j != col
                ]
                for i in range(self.row)
                if i != row
            ])

    def minor(self, row, col):
        u"""小行列式"""
        return self.sub(row, col).det()

    def __getitem__(self, key):
        if type(key) is tuple:
            if len(key) != 2:
                raise IndexError('If you access the element, length of indices should be 2')
            if type(key[0]) is not int or type(key[0]) != type(key[1]):
                raise TypeError(
                    'Type of index should be int\n'
                    'Slicing is not supported yet...')

            return self.data[key[0]][key[1]]
        elif type(key) is int:
            return self.data[key]
        else:
            raise TypeError(
                'Type of index should be int\n'
                'Slicing is not supported yet...')

    def __repr__(self):
        return self.__str__()

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
        u"""全要素をそれぞれ除算する

        なお，0除算エラーはpythonの挙げる例外に身を任せる
        """
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
