# -*- coding: utf-8 -*-
import matplotlib.pyplot as pyplot
from matrix_decomposition_from_scratch.matrices import Matrix


class Historic2D(object):
    def __init__(self, init_point_2d):
        u"""初期位置の設定
        init_point_2d: 初期位置(Matrix)
        """
        if type(init_point_2d) is not Matrix:
            raise TypeError('Initial point should be Matrix')
        if init_point_2d.col != 1 or init_point_2d.row != 2:
            raise IndexError('Invalid shape')
        self._traj_2d = [init_point_2d]

        # 描画器の準備
        self._fig = pyplot.figure('Trajectory 2D')
        self._ax = self._fig.add_subplot(1, 1, 1)

    def transform(self, mat):
        u"""与えられた行列で最新の位置を線形変換し，リストに追加する"""
        new_point_2d = mat.dot(self._traj_2d[-1])
        self._traj_2d.append(new_point_2d)

    def plot(self):
        u"""保持している奇跡のプロット

        プロットできる形式に変換してから描画処理をする
        """
        traj_x = [mat[0, 0] for mat in self._traj_2d]
        traj_y = [mat[1, 0] for mat in self._traj_2d]
        print(traj_x)
        print(traj_y)

        self._fig.clf()
        self._ax.cla()
        self._fig = pyplot.figure('Trajectory 2D')
        self._ax = self._fig.add_subplot(1, 1, 1)

        self._ax.plot(traj_x, traj_y, marker='.')
        pyplot.pause(0.01)
