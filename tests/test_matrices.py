# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy.linalg as LA
import random
from sys import float_info
from unittest import TestCase

from matrix_decomposition_from_scratch.matrices import Matrix
import matrix_decomposition_from_scratch.matrices as matrices

class MatrixTests(TestCase):

    def _compare_all_elements(self, data, model):
        for r in range(data.row):
            for c in range(data.col):
                self.assertAlmostEqual(data[r, c], model[r, c])

    def _generate_random_list(self, a=-1., b=1., row=None, col=None):
        if row is None or col is None:
            row = math.ceil(random.random() * 10)
            col = math.ceil(random.random() * 10)
        return \
            [
                [
                    random.uniform(a, b)
                    for i in range(col)
                ]
                for j in range(row)
            ]

    def test_init_valid(self):
        # setup
        test_list = self._generate_random_list()
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise and verify
        self._compare_all_elements(test_mat, np_mat)


    def test_shape(self):
        # setup
        test_list = self._generate_random_list()
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        t_row = test_mat.row
        t_col = test_mat.col

        # verify
        self.assertEqual(t_row, np_mat.shape[0])
        self.assertEqual(t_col, np_mat.shape[1])

    def test_tolist(self):
        # setup
        test_list = self._generate_random_list()
        test_mat = Matrix(test_list)

        # exercise
        test_tolist = test_mat.tolist()

        # verify
        self.assertEqual(test_list, test_tolist)

    def test_transpose(self):
        # setup
        test_list = self._generate_random_list()
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        transposed_test_mat = test_mat.transpose()
        transposed_np_mat = np_mat.transpose()

        # verify
        self._compare_all_elements(transposed_test_mat, transposed_np_mat)

    def test_trance(self):
        # setup
        test_list = self._generate_random_list()
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        trace_test = test_mat.trace()
        trace_np = np_mat.trace()

        self.assertAlmostEqual(trace_test, trace_np)

    def test_dot_invalid(self):
        # setup
        test_list = self._generate_random_list(row=3, col=4)
        test_mat = Matrix(test_list)

        # exercise and verify
        with self.assertRaises(ValueError):
            test_mat.dot(test_mat)

    def test_dot_valid(self):
        # setup
        test_list = self._generate_random_list()
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        test_dot = test_mat.dot(test_mat.transpose())
        np_dot = np_mat.dot(np_mat.transpose())

        # verify
        self._compare_all_elements(test_dot, np_dot)
        self.assertEqual(test_dot.row, test_dot.col)

    def test_det_invalid(self):
        # setup
        test_list = self._generate_random_list(row=3, col=4)
        test_mat = Matrix(test_list)

        # exercise and verify
        with self.assertRaises(ValueError):
            test_mat.det()

    def test_det_valid(self):
        # setup
        test_list = self._generate_random_list(row=5, col=5)
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        test_det = test_mat.det()
        np_det = LA.det(np_mat)

        # verify
        self.assertAlmostEqual(test_det, np_det, places=3)

    def test_inv_invalid_shape(self):
        # setup
        test_list = self._generate_random_list(row=3, col=4)
        test_mat = Matrix(test_list)

        # exercise and verify
        with self.assertRaises(ValueError):
            test_mat.inv()

    def test_inv_invalid_value(self):
        # setup
        test_list = \
            [
                [
                    j + 5 * i
                    for j in range(5)
                ]
                for i in range(5)
            ]
        test_mat = Matrix(test_list)

        # exercise and verify
        with self.assertRaises(ValueError):
            test_mat.inv()

    def test_inv_invalid_value(self):
        # setup
        test_list = \
            [
                [1., 2., 3., 4.],
                [8., 7., 6., 8.],
                [9., 12., 11., 12.],
                [16., 15., 14., 13.],
            ]
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        test_inv = test_mat.inv()
        np_inv = LA.inv(np_mat)

        # verify
        self._compare_all_elements(test_inv, np_inv)

    def test_matmul(self):
        # setup
        test_list = self._generate_random_list()
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        test_dot = test_mat @ test_mat.transpose()
        np_dot = np_mat @ np_mat.transpose()

        # verify
        self._compare_all_elements(test_dot, np_dot)
        self.assertEqual(test_dot.row, test_dot.col)

    def test_eye(self):
        # setup
        d = 3

        # exercise
        test_eye = matrices.eye(d)
        np_eye = np.eye(d)

        # verify
        self._compare_all_elements(test_eye, np_eye)
        self.assertEqual(test_eye.row, test_eye.col)

    def test_argmax(self):
        # setup
        test_list = self._generate_random_list()
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        test_arg = matrices.argmax(test_mat)
        np_arg = np.argmax(np_mat.reshape(1, np_mat.shape[0] * np_mat.shape[1]))

        # verify
        self.assertEqual(test_arg, np_arg)

    def test_abs(self):
        # setup
        test_list = self._generate_random_list()
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        test_abs = matrices.abs(test_mat)
        np_abs = np.abs(np_mat)

        # verify
        self._compare_all_elements(test_abs, np_abs)
        self.assertEqual(test_mat.row, np_mat.shape[0])
        self.assertEqual(test_mat.col, np_mat.shape[1])

    def test_reshape(self):
        # setup
        row_list = [1, 8, 4,]
        col_list = [16, 2, 4]
        test_list = self._generate_random_list(row=2, col=8)
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise and verify
        for to_row, to_col in zip(row_list, col_list):
            test_reshape = test_mat.reshape(to_row, to_col)
            np_reshape = np_mat.reshape(to_row, to_col)

            self._compare_all_elements(test_reshape, np_reshape)
            self.assertEqual(test_reshape.row, np_reshape.shape[0])
            self.assertEqual(test_reshape.col, np_reshape.shape[1])

    def test_max(self):
        # setup
        test_list = self._generate_random_list(row=4, col=3)
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        test_max = matrices.max(test_mat)
        np_max = np.max(np_mat)

        # verify
        self.assertEqual(np_max, test_max)

    def test_eigen_jacob(self):
        # setup
        test_list = [
            [[1, 2, 3],
             [2, 4, 5],
             [3, 5, 1]],
            [[1, 2, 3, 4],
             [2, 5, 4, 0],
             [3, 4, 1, 1],
             [4, 0, 1, 2]]]
        # Numpyの結果とベクトルの並びが違うので，参考ソースとの結果を比較する
        # Rigen values and vectors have no restricted alignment.
        # So I compare my result with reference code's one.
        expected_list = [
            {
                'val': np.array([9.15554844, 0.03424156, -3.18979]),
                'vec': np.array([[0.3890768121223604, 0.8484624273927595 , -0.3587906681778219],
                                 [0.7146655786221864, -0.523764686501912 , -0.4635986021400077],
                                 [0.5812678771455694, 0.07603987425252731, 0.8101515861382627]])
            },
            {
                'val': np.array([9.584292205173988, 4.243778959253601, -1.5548070077212375, -3.2732641567063494]),
                'vec': np.array(
                    [[0.47493891922778086, 0.41787359408281666, 0.18749368737154454,  0.7514391522717447],
                     [0.6475162083223736,  -0.5703677880234859, -0.5042468744699627,  0.03374071623848818],
                     [0.5046314170201278,  -0.101105766474563,  0.7325633929135483,   -0.44550604070521777],
                     [0.31702194863892624, 0.699885611896042,   -0.41705165405029304, -0.48551532642749406]])
            }]

        # exercise and verify
        for mat, expected in zip(test_list, expected_list):
            test_mat = Matrix(mat)

            # exercise
            test_val, test_vec = test_mat.eigen(method='jacob')

            # verify
            self.assertEqual(len(expected['val']), len(test_val))
            for v, e in zip(test_val, expected['val']):
                self.assertAlmostEqual(v, e)

            self.assertEqual(expected['vec'].shape[0], test_vec.row)
            self.assertEqual(expected['vec'].shape[1], test_vec.col)
            for r in range(test_vec.row):
                for c in range(test_vec.col):
                    self.assertAlmostEqual(expected['vec'][r, c], test_vec[r, c])
            # VVinv = I
            V = test_vec
            I = matrices.eye(V.row)
            for r in range(V.row):
                for c in range(V.col):
                    self.assertAlmostEqual(I[r, c], (V @ V.inv())[r, c])
            # VSVinv = A
            S = matrices.diag(test_val)
            for r in range(test_mat.row):
                for c in range(test_mat.col):
                    self.assertAlmostEqual(test_mat[r, c], (V @ S @ V.inv())[r, c])

    def test_diag_mat(self):
        # setup
        test_list = self._generate_random_list(row=4, col=3)
        test_mat = Matrix(test_list)
        np_mat = np.array(test_list)

        # exercise
        test_diag = test_mat.diag()
        np_diag = np.diag(np_mat)

        # verify
        self._compare_all_elements(test_diag, np.matrix(np_diag))
