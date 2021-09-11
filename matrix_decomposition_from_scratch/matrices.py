# -*- coding: utf-8 -*-
from copy import deepcopy
from sys import float_info
import math
from math import cos, sin

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
        if math.fabs(self.det()) < float_info.epsilon:
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

    def eigen(self, method='jacob'):
        u"""Compute eigen vector and values

        Note: Row major eigen vector
        """
        if self.row != self.col:
            raise ValueError('Eigenvalues decomposition is defined for square matrix')

        methods = {
            'jacob': self._eigen_jacob,
            'qr': self._eigen_qr,
            'power': self._eigen_power
            }
        vals, vec = methods[method]()
        # sorting
        # https://takumaro-blog.com/pc関連/【python】固有値分解・特異値分解のソート並び替え/
        # TODO(fugashy) Implement by self
        from numpy import argsort, array, matrix
        npvals = array(vals)
        index = argsort(vals)[::-1]

        ret_vals = npvals[index].tolist()
        ret_vec = Matrix(matrix(vec._data).T[index].tolist()).transpose()

        return ret_vals, ret_vec

    def svd(self):
        u""" Singular value decomposition

        https://ohke.hateblo.jp/entry/2017/12/14/230500
        """
        A = self.transpose() @ self
        evals, evecs = A.eigen(method='jacob')

        svals = [math.sqrt(e) for e in evals]

        Z = diag(svals)
        V = evecs
        U = Matrix(
            [
                ((self @ Matrix([V.transpose()[i]]).transpose()) / svals[i]).transpose().tolist()[0]
                for i in range(len(svals))
            ]).transpose()

        return U, Z, V

    def mean(self, axis=None):
        # 総和
        if axis is None:
            return sum(self.reshape(1, self.row * self.col).tolist()[0]) / (self.row * self.col)
        # 行毎
        elif axis == 1:
            return Matrix([
                    [sum(self[r]) / self.col]
                    for r in range(self.row)
                ])

    def _eigen_jacob(self):
        u"""
        https://ensekitt.hatenablog.com/entry/2018/07/19/200000
        """
        inveye = -1.0 * (eye(self.row) - 1)
        tol = 1e-10

        R = eye(self.row)

        X = deepcopy(self)
        for i in range(100):
            # 最大値の抽出
            absval = abs(X * inveye).reshape(1, X.row * X.col)
            amax = max(absval)
            if amax < 1e-10:
                break

            aindex = argmax(absval)
            r = int(aindex / X.row)
            c = int(aindex % X.row)

            if X[r, r] - X[c, c] == 0:
                theta = 0.25 * math.pi
            else:
                theta = 0.5 * math.atan(-2.0 * X[r, c] / (X[r, r] - X[c, c]))

            new_X = deepcopy(X)
            for k in range(X.row):
                new_X[r, k] = X[r, k] * cos(theta) - X[c, k] * sin(theta)
                new_X[k, r] = new_X[r, k]
                new_X[c, k] = X[r, k] * sin(theta) + X[c, k] * cos(theta)
                new_X[k, c] = new_X[c, k]

            new_X[r, r] = \
                (X[r, r] + X[c, c]) / 2. + \
                (X[r, r] - X[c, c]) / 2. * cos(2 * theta) - \
                X[r, c] * sin(2 * theta)
            new_X[c, c] = \
                (X[r, r] + X[c, c]) / 2. - \
                (X[r, r] - X[c, c]) / 2. * cos(2 * theta) + \
                X[r, c] * sin(2 * theta)
            new_X[r, c] = 0.
            new_X[c, r] = 0.
            X = deepcopy(new_X)

            G = eye(X.row)
            G[r, r] = cos(theta)
            G[r, c] = sin(theta)
            G[c, r] = -sin(theta)
            G[c, c] = cos(theta)
            R = R @ G

        # col of R is one of eigen vector
        return diag(X)[0], R

    def _eigen_qr(self):
        raise NotImplementedError

    def _eigen_power(self):
        u"""
        https://qiita.com/sci_Haru/items/e5278b45ab396424ad86
        """
        raise NotImplementedError

    def reshape(self, r, c):
        if r <= 0 or c <= 0:
            raise ValueError('r and c must be larger than 0 to reshape mat')

        vec = [self._data[ir][ic] for ir in range(self.row) for ic in range(self.col)]
        return Matrix(
            [
                [
                    vec[ir * c + ic]
                    for ic in range(c)
                ]
                for ir in range(r)
            ])

    def diag(self):
        return diag(self)

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

    def __setitem__(self, key, value):
        if type(key) is tuple:
            if len(key) != 2:
                raise IndexError('If you access the element, length of indices should be 2')
            if type(key[0]) is not int or type(key[0]) != type(key[1]):
                raise TypeError(
                    'Type of index should be int\n'
                    'Slicing is not supported yet...')
            if type(value) is not int and type(value) is not float:
                raise TypeError('value to be set must be scalar')
            self._data[key[0]][key[1]] = value

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

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        u"""行列積を行う"""
        return self.dot(other)

    def __rmatmul__(self, other):
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

def argmax(mat):
    max_val = -float_info.min
    r = 0
    c = 0
    for ir in range(mat.row):
        for ic in range(mat.col):
            if mat[ir, ic] > max_val:
                max_val = mat[ir, ic]
                r = ir
                c = ic
    return r * mat.col + c

def abs(mat):
    return Matrix(
        [
            [
                math.fabs(mat[ir, ic])
                for ic in range(mat.col)
            ]
            for ir in range(mat.row)
        ])

def max(mat):
    max_val = -float_info.min
    for ir in range(mat.row):
        for ic in range(mat.col):
            if mat[ir, ic] > max_val:
                max_val = mat[ir, ic]
    return max_val

def diag(obj):
    if type(obj) is Matrix:
        values = list()
        for r, c in zip(range(obj.row), range(obj.col)):
            values.append(obj[r, c])
        return Matrix([values])
    elif type(obj) is list:
        return Matrix(
            [
                [
                    obj[i] if i == j else 0.
                    for i in range(len(obj))
                ]
                for j in range(len(obj))
            ])
        raise NotImplementedError

def outer(a, b):
    if type(a) is not Matrix or type(b) is not Matrix:
        raise TypeError('outer method requres Matrix')

    return a.reshape(a.col * a.row, 1) @ b.reshape(1, b.col * b.row)


