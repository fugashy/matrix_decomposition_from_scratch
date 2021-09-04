# -*- coding: utf-8 -*-
import sys
sys.path.append('../matrix_decomposition_from_scratch')
from matplotlib import cm
import matplotlib.pyplot as pyplot
from matrices import Matrix
import numpy as np


class Historic2D(object):
    def __init__(self, init_points):
        u"""初期位置の設定
        init_points: 初期位置(list(Matrix))
        """
        if type(init_points) is not list:
            raise TypeError('Initial point should be list of Matrix')
        for init_point in init_points:
            if type(init_point) is not Matrix:
                raise TypeError('Initial point should be list of Matrix')
            if init_point.col != 1 or init_point.row != 2:
                raise IndexError('Invalid shape point')

        self._dists = [init_points]

        self._reset_viewer()

    def transform(self, mat):
        u"""与えられた行列で最新の位置を線形変換し，リストに追加する"""
        new_points = list()
        for point in self._dists[-1]:
            new_points.append(mat @ point)
        self._dists.append(new_points)

    def plot(self):
        u"""保持している奇跡のプロット

        プロットできる形式に変換してから描画処理をする
        """
        self._reset_viewer()

        for i, dist in enumerate(self._dists):
            xs = [p[0, 0] for p in dist]
            ys = [p[1, 0] for p in dist]

            self._ax.scatter(xs, ys, label=str(i), marker='.', alpha=0.5, color=cm.jet(i / len(self._dists)))
        pyplot.pause(0.01)

    def _reset_viewer(self):
        self._fig = pyplot.figure('Points')
        self._ax = self._fig.add_subplot(1, 1, 1, aspect='equal')

