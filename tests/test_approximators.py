from unittest import TestCase
import numpy as np

from structurednets.approximators.psm_approximator import PSMApproximator
from structurednets.approximators.psm_approximator_wrapper import PSMApproximatorWrapper
from structurednets.approximators.sss_approximator import SSSApproximator
from structurednets.approximators.ldr_approximator import LDRApproximator
from structurednets.approximators.hodlr_approximator import HODLRApproximator
from structurednets.approximators.lr_approximator import LRApproximator

class LayerTests(TestCase):
    def test_psm_approximator(self):
        nnz_share = 0.2
        optim_mat = np.random.uniform(-1,1, size=(51,10))
        approximator = PSMApproximator(nb_matrices=2, linear_nb_nonzero_elements_distribution=True)
        res = approximator.approximate(optim_mat, nb_params_share=nnz_share)
        approx_mat_dense = res["approx_mat_dense"]
        self.assertTrue(np.array_equal(approx_mat_dense.shape, np.array([51, 10])), "The approximated optim_mat has the wrong shape")
        self.assertTrue(np.sum([np.sum(sparse_factor != 0) for sparse_factor in res["faust_approximation"]]) <= int(nnz_share * optim_mat.size), " Too many non zero elements")

    def test_psm_approximator_wrapper(self):
        nnz_share = 0.2
        optim_mat = np.random.uniform(-1,1, size=(51,10))
        approximator = PSMApproximatorWrapper()
        res = approximator.approximate(optim_mat, nb_params_share=nnz_share)
        approx_mat_dense = res["approx_mat_dense"]
        self.assertTrue(np.array_equal(approx_mat_dense.shape, np.array([51, 10])), "The approximated optim_mat has the wrong shape")
        self.assertTrue(np.sum([np.sum(sparse_factor != 0) for sparse_factor in res["faust_approximation"]]) <= int(nnz_share * optim_mat.size), " Too many non zero elements")

    def test_sss_approximator(self):
        optim_mat = np.random.uniform(-1,1, size=(20,16))
        approximator = SSSApproximator(nb_states=4)
        res = approximator.approximate(optim_mat, nb_params_share=0.45)
        approx_mat_dense = res["approx_mat_dense"]
        self.assertTrue(np.array_equal(approx_mat_dense.shape, np.array([20,16])), "The approximated optim_mat has the wrong shape")

    def test_ldr_approximator(self):
        optim_mat = np.random.uniform(-1,1, size=(20, 20))
        approximator = LDRApproximator()
        res = approximator.approximate(optim_mat, nb_params_share=0.2)
        approx_mat_dense = res["approx_mat_dense"]
        self.assertTrue(np.array_equal(approx_mat_dense.shape, np.array([20, 20])), "The approximated optim_mat has the wrong shape")

    def test_lr_approximator(self):
        optim_mat = np.random.uniform(-1,1, size=(28, 20))
        param_share = 0.2
        max_nb_parameters = int(optim_mat.size * param_share)

        approximator = LRApproximator()
        res = approximator.approximate(optim_mat, nb_params_share=param_share)

        approx_mat_dense = res["approx_mat_dense"]
        nb_parameters = res["nb_parameters"]
        self.assertTrue(np.array_equal(approx_mat_dense.shape, np.array([28, 20])), "The approximated optim_mat has the wrong shape")
        self.assertTrue(nb_parameters <= max_nb_parameters, "The number of parameters in the approximation should be lower or equal the maximum number of allowed parameters")

        lr_left_size = np.random.uniform(-1 ,1, size=(26, 4))
        lr_right_side = np.random.uniform(-1, 1, size=(4, 17))
        lr_optim_mat = lr_left_size @ lr_right_side
        param_share = (lr_left_size.size + lr_right_side.size) / lr_optim_mat.size

        res = approximator.approximate(lr_optim_mat, nb_params_share=param_share)
        approx_mat_dense = res["approx_mat_dense"]
        self.assertTrue(np.allclose(lr_optim_mat, approx_mat_dense), "The approximator should be able to find the structure of a low rank matrix")

    def test_hodlr_approximator(self):
        nnz_share = 0.2
        optim_mat = np.random.uniform(-1,1, size=(51,10))
        approximator = HODLRApproximator()
        res = approximator.approximate(optim_mat, nb_params_share=nnz_share)
        approx_mat_dense = res["approx_mat_dense"]
        self.assertTrue(np.array_equal(approx_mat_dense.shape, np.array([51, 10])), "The approximated optim_mat has the wrong shape")

    def test_hodlr_with_hodlr_matrix(self):
        rand_mat = np.random.uniform(-1,1, size=(47,34))
        approximator_1 = HODLRApproximator()
        res = approximator_1.approximate(rand_mat, nb_params_share=0.2)
        optim_mat = res["approx_mat_dense"]

        approximator_2 = HODLRApproximator()
        res = approximator_2.approximate(optim_mat, nb_params_share=0.2)
        approx_mat_dense = res["approx_mat_dense"]
        self.assertTrue(np.allclose(optim_mat, approx_mat_dense), "The HODLR approximation algorithm should be able to fully recover a HODLR matrix")