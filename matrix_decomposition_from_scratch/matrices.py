# -*- coding: utf-8 -*-
from sys import float_info

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
        self._data = \
            [
                [
                    float(value)
                    for value in row_data
                ]
                for row_data in data
            ]

    u"""-----------------------Properties-----------------------------------"""
    @property
    def row(self):
        return len(self._data)

    @property
    def col(self):
        return len(self._data[0])

    u"""-----------------------Public methods-------------------------------"""
    def tolist(self):
        return self._data

    def transpose(self):
        u"""転置行列を返す"""
        return Matrix(
            [
                [
                    self[c, r]
                    for c in range(self.row)
                ]
                for r in range(self.col)
            ])

    def trace(self):
        u"""トレースを返す

        対角成分の総和
        """
        trace_id = min([self.col, self.row])
        return sum([self[i, i] for i in range(trace_id)])

    def dot(self, other):
        u"""InnerProductを計算する"""
        # 同じ型同士でなければならない
        if not isinstance(other, Matrix):
            raise TypeError('dot function requires object type of Matrix')

        # 左の列と右の行が一致しているか
        if self.col != other.row:
            raise ValueError('dot requires the condiction self.col == self.row')

        return Matrix(
            [
                [
                    sum(
                        [
                            self[i, k] * other[k, j]
                            for k in range(self.col)
                        ])
                    for j in range(other.col)
                ]
                for i in range(self.row)
            ])

    def det(self):
        u"""行列式を計算する

        正方行列にのみ定義された量である

        3x3まではサラスの公式を使う
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
            sum = 0.0
            for i in range(self.row):
                sum += self.cofactor(i, 0) * self[i, 0]
            return sum

    def inv(self):
        u"""逆行列を計算する"""
        determinant = self.det()
        if abs(self.det()) < float_info.epsilon:
            raise ValueError('Self determinant is almost zero')

        return Matrix(
            [
                [
                    self.cofactor(r, c)
                    for r in range(self.row)
                ]
                for c in range(self.col)
            ]) / determinant

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

    def cofactor(self, row, col):
        u"""余因子を計算する"""
        return pow(-1., row + col + 2) * self.minor(row, col)

    TRI_UPPER=0
    TRI_LOWER=1

    def triangular(self, mode=TRI_UPPER, include_diagonals=True):
        u"""三角行列を返す

        上か下か
        対角成分を含むか含まないか

        以上を設定できる
        デフォルト設定は，対角成分を含む上三角行列
        """
        ret = Matrix([[ 0.0 for c in range(self.col)] for r in range(self.row)])
        compare_index = lambda r, c: c >= r
        if mode == self.TRI_UPPER and not include_diagonals:
            compare_index = lambda r, c: c > r
        elif mode == self.TRI_LOWER and include_diagonals:
            compare_index = lambda r, c: c <= r
        elif mode == self.TRI_LOWER and not include_diagonals:
            compare_index = lambda r, c: c < r

        for r in range(ret.row):
            for c in range(ret.col):
                if compare_index(r, c):
                    ret._data[r][c] = self[r, c]
        return ret

    u"""-----------------------Overloads------------------------------------"""
    def __getitem__(self, key):
        if type(key) is tuple:
            if len(key) != 2:
                raise IndexError('If you access the element, length of indices should be 2')
            if type(key[0]) is not int or type(key[0]) != type(key[1]):
                raise TypeError(
                    'Type of index should be int\n'
                    'Slicing is not supported yet...')

            return self._data[key[0]][key[1]]
        elif type(key) is int:
            return self._data[key]
        else:
            raise TypeError(
                'Type of index should be int\n'
                'Slicing is not supported yet...')

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        ret_str = '['
        for ir, row_data in enumerate(self._data):
            if ir == 0:
                ret_str += '[ '
            else:
                ret_str += ' [ '

            for cols_value in row_data:
                ret_str += str(cols_value) + ' '

            if ir != len(self._data) - 1:
                ret_str += ']\n'
            else:
                ret_str += ']'
        ret_str += ']'
        return ret_str

    def __add__(self, other):
        u"""全要素をそれぞれ加算する"""
        if isscalar(other):
            return Matrix(self._calc_all_elements(
                other, lambda l, r: l + r))
        self._verificate_shape_is_same(other)
        return Matrix(self._calc_all_elements_of_matrix(
            other, lambda l, r: l + r))

    def __sub__(self, other):
        u"""全要素をそれぞれ減算する"""
        if isscalar(other):
            return Matrix(self._calc_all_elements(
                other, lambda l, r: l - r))
        self._verificate_shape_is_same(other)
        return Matrix(self._calc_all_elements_of_matrix(
            other, lambda l, r: l - r))

    def __mul__(self, other):
        u"""全要素をそれぞれ乗算する"""
        if isscalar(other):
            return Matrix(self._calc_all_elements(
                other, lambda l, r: l * r))
        self._verificate_shape_is_same(other)
        return Matrix(self._calc_all_elements_of_matrix(
            other, lambda l, r: l * r))

    def __matmul__(self, other):
        u"""行列積を行う"""
        return self.dot(other)

    def __truediv__(self, other):
        u"""全要素をそれぞれ除算する

        なお，0除算エラーはpythonの挙げる例外に身を任せる
        """
        if isscalar(other):
            return Matrix(self._calc_all_elements(
                other, lambda l, r: l / r))
        self._verificate_shape_is_same(other)
        return Matrix(self._calc_all_elements_of_matrix(
            other, lambda l, r: l / r))

    u"""-----------------------Private methods------------------------------"""
    def _verificate_shape_is_same(self, other):
        if self.col != other.col or self.row != other.row:
            raise ValueError('Shape of both should be same')

    def _calc_all_elements_of_matrix(self, other, operator):
        return \
            [
                [
                    operator(self._data[ir][ic], other._data[ir][ic])
                    for ic in range(self.col)
                ]
                for ir in range(self.row)
            ]

    def _calc_all_elements(self, o, operator):
        return \
            [
                [
                    operator(self[ir, ic], o)
                    for ic in range(self.col)
                ]
                for ir in range(self.row)
            ]

def isscalar(o):
    return type(o) is int or type(o) is float

def eye(d):
    if d < 2:
        raise ValueError('d must be larger than 2 for generate eye matrix')

    return Matrix(
        [
            [
                1 if i == j else 0.
                for i in range(d)
            ]
            for j in range(d)
        ])
