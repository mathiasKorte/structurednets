import numpy as np
from sklearn.cluster import AgglomerativeClustering
from structurednets.approximators.approximator import Approximator
from structurednets.hmatrix.block_cluster_tree import BlockClusterTree
from structurednets.hmatrix.tree_element import TreeElement
from structurednets.hmatrix.hmatrix import HMatrix
from structurednets.models.googlenet import GoogleNet

def permuate_clusters(M, depth=5):
    def get_even_clusters(X):     
        # distance_matrix = np.array([[(X[i,i]+X[j,j]-2*X[i,j]) for i in range(X.shape[0])] for j in range(X.shape[0])])
        distance_matrix = np.square([[np.linalg.norm(X[i,:]-X[j,:]) for i in range(X.shape[0])] for j in range(X.shape[0])])
        clusters = AgglomerativeClustering(n_clusters=2, metric='precomputed',linkage='complete').fit_predict(distance_matrix)
        for idx in np.argsort(distance_matrix[np.where(clusters == 1)[0][0],:]):
            if(np.sum(clusters==0)<np.sum(clusters==1)+1):
                break
            clusters[idx]=1
        return clusters

    for d in range(depth):
            block_num = int(2**d)
            xd = (M.shape[0]/block_num)
            yd = (M.shape[1]/block_num)

            x_perm_list=[]
            y_perm_list=[]
            for b in range(block_num):

                if (b+1<block_num):
                    B = M[int(b*xd):int((b+1)*xd), int(b*yd):int((b+1)*yd)]
                else:
                    B = M[int(b*xd):, int(b*yd):]

                x_part_perm = np.argsort(get_even_clusters(B))
                y_part_perm = np.argsort(get_even_clusters(B.T))             
                
                x_part_perm += int(b*xd)
                y_part_perm += int(b*yd)

                x_perm_list.append(x_part_perm)
                y_perm_list.append(y_part_perm)

            x_perm = np.concatenate(x_perm_list)
            y_perm = np.concatenate(y_perm_list) 

            M = M[x_perm,:]
            M = M[:,y_perm]
    return M

def build_hodlr_block_cluster_tree(depth: int, matrix_shape: tuple, min_block_size=2) -> BlockClusterTree:
    root = TreeElement(children=None, row_range=range(matrix_shape[0]), col_range=range(matrix_shape[1]))
    res = BlockClusterTree(root=root)
    for _ in range(1, depth):
        res.split_hodlr_style(matrix_shape=matrix_shape, min_block_size=min_block_size)
    res.check_validity(matrix_shape=matrix_shape)
    return res

class HODLRApproximator(Approximator):
    def approximate(self, optim_mat: np.ndarray, nb_params_share: float, max_depth=5):
        best_hmatrix = None
        best_hmatrix_error = None

        for depth in range(1, max_depth):
            block_cluster_tree = build_hodlr_block_cluster_tree(depth=depth, matrix_shape=optim_mat.shape)
            hmatrix = HMatrix(block_cluster_tree=block_cluster_tree)
            hmatrix.find_best_leaf_approximation(optim_mat=optim_mat, nb_params_share=nb_params_share)

            curr_error = hmatrix.get_curr_error(optim_mat=optim_mat)
            if best_hmatrix_error is None \
                or curr_error < best_hmatrix_error:
                best_hmatrix_error = curr_error
                best_hmatrix = hmatrix

        best_hmatrix.clear_full_rank_parts_and_cached_values()
        res_dict = dict()
        res_dict["type"] = "HODLRApproximator"
        res_dict["h_matrix"] = best_hmatrix
        res_dict["approx_mat_dense"] = best_hmatrix.to_dense_numpy()
        res_dict["nb_parameters"] = best_hmatrix.block_cluster_tree.get_nb_params()
        return res_dict

    def get_name(self):
        return "HODLRApproximator"

if __name__ == "__main__":
    # tree = build_hodlr_block_cluster_tree(8, (100, 100))
    # tree.plot()

    # model = GoogleNet(output_indices=np.arange(1000), use_gpu=False)
    # optim_mat = model.get_optimization_matrix().detach().numpy()

    optim_mat = np.random.rand(100,100)    

    approximator = HODLRApproximator()
    res = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.4)
    print("Approximation Error: " + str(np.linalg.norm(res["approx_mat_dense"] - optim_mat, ord="fro")))

    print("permuting based on clustering...")
    optim_mat = permuate_clusters(optim_mat)

    approximator = HODLRApproximator()
    res = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.4)
    print("Approximation Error: " + str(np.linalg.norm(res["approx_mat_dense"] - optim_mat, ord="fro")))

