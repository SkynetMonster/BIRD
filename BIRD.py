import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import os
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter

from Real_Gen.arfpy import arf
# @misc{blesch2023arfpypythonpackagedensity,
#       title={arfpy: A python package for density estimation and generative modeling with adversarial random forests}, 
#       author={Kristin Blesch and Marvin N. Wright},
#       year={2023},
#       eprint={2311.07366},
#       archivePrefix={arXiv},
#       primaryClass={stat.ML},
#       url={https://arxiv.org/abs/2311.07366}, 
# }
from Real_Gen.hard_decision import lim_cluster_tabular_data
from ucimlrepo import fetch_ucirepo 

from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_statistical import JensenShannonDistance, InverseKLDivergence
from Metrics.ppr import compute_pprecision_precall

# Load the scoringRules package
ro.r('library(scoringRules)')
ro.r('library(mlbench)')


class RBF:
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, X):
        pairwise_dists = torch.cdist(X, X).pow(2)
        return torch.exp(-self.gamma * pairwise_dists)

class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY



# Support Functions
def rbf_median(X):
    pairwise_dists = torch.cdist(X, X)
    median_dist = torch.median(pairwise_dists[pairwise_dists != 0])
    return 1 / (2 * median_dist ** 2)

def compute_similarity(true, pred, similarity_measure='new_es'):
    true_np = true.detach().numpy()
    pred_np = pred.detach().numpy()
    if similarity_measure == "energy_score":
        return 1 / compute_energy_score_r(pred_np, true_np)
        #return 1 / sr.energy_score(true_np, pred_np)
    elif similarity_measure == "new_es":
        return 1 / energy_score(true_np, pred_np)

def compute_energy_score_r(predictions, observations):
    """
    Calls R scoringRules::es_sample() from Python
    """

    with localconverter(default_converter + numpy2ri.converter):
        r_predictions = ro.conversion.py2rpy(predictions)
        r_observations = ro.conversion.py2rpy(ro.FloatVector(observations))

    r_predictions = ro.r('t')(r_predictions)

    energy_score = ro.r("es_sample")(y=r_observations, dat=r_predictions)

    with localconverter(default_converter + numpy2ri.converter):
        result = ro.conversion.rpy2py(energy_score)

    return result[0]

def energy_score(y, X):
    """
    Computes the energy score for a given observation and predictive samples.

    Parameters:
    - y: ndarray of shape (d,), observed value
    - X: ndarray of shape (M, d), predictive samples

    Returns:
    - Energy score (float)
    """
    X = np.atleast_2d(X)
    y = np.atleast_1d(y)
    M = X.shape[0]

    # First term: average L2 distance between X and y
    d1 = np.mean(np.linalg.norm(X - y, axis=1))

    # Second term: average pairwise L2 distance among predictive samples
    pairwise_mean = np.mean(pdist(X, metric='euclidean'))
    d2 = (M - 1) / M * pairwise_mean

    return d1 - 0.5 * d2

def metrics_evaluation(synthetic_data, real_data):
    # Assume both are tensors
    if not isinstance(synthetic_data, pd.DataFrame):
        synthetic_data_copy = pd.DataFrame(synthetic_data.detach().numpy())

    if not isinstance(real_data, pd.DataFrame):
        real_data_copy = pd.DataFrame(real_data.detach().numpy())

    # Initialize the data loaders
    synthetic_data_loader = GenericDataLoader(synthetic_data_copy)
    real_data_loader = GenericDataLoader(real_data_copy)

    # Initialize the metrics
    mmd = MMDLoss(kernel=RBF(gamma=mh_gamma))
    jsd = JensenShannonDistance()
    ikl = InverseKLDivergence()

    # Initialize a DataFrame to store results
    results = pd.DataFrame(index=['Mixture'], columns=["MMD", "JSD", "IKL", "P-PR", "P-RE"])
    # Calculate metrics for each generated dataset
    mmd_distance = mmd(synthetic_data, real_data)
    jsd_distance = jsd.evaluate(synthetic_data_loader, real_data_loader)
    ikl_distance = ikl.evaluate(synthetic_data_loader, real_data_loader)
    ppr, pre = compute_pprecision_precall(real_data.detach().numpy(), synthetic_data.detach().numpy())

    # Convert the results to a dictionary
    mmd_result = np.round(mmd_distance.item(),7)
    jsd_result = np.round(list(jsd_distance.values())[0],7)
    ikl_result = np.round(list(ikl_distance.values())[0],7)
    den_result = np.round(ppr,7)
    cov_result = np.round(pre,7)

    # Store the result in the DataFrame
    results.iloc[0] = [mmd_result, jsd_result, ikl_result, den_result, cov_result]
    return results


# Main Functions
def bandit(time_dict, cluster, list_of_dist, bandit_type='R-SR', n_samples=1000, n_features=2, budget=20, window_size=0.25, explore_para=1, discrete_columns=[], epsilon = 1, mh_gamma=1.0):
    n_components = len(list_of_dist)
    list_of_models = []
    for i in range(n_components):
        if list_of_dist[i] == 'ARF':
            list_of_models.append(arf.arf(epochs=1))
        elif list_of_dist[i] == ...:
            list_of_models.append(...)
        else:
            raise ValueError(f"Unknown distribution name: {list_of_dist[i]}")

    lc_gamma = rbf_median(cluster)

    if bandit_type == 'R-UCBE':
        return ...
    
    elif bandit_type == 'R-SR':
        def f_N(j, log_k):
            if j == 0:
                return 0
            else:
                return int(torch.ceil(torch.tensor((budget - n_components) / (log_k * (n_components + 1 - j)))))

        log_k = sum([1 / i for i in range(1, n_components + 1)]) + 1 / 2
        k_list = [i for i in range(n_components)]

        for j in range(1, n_components):
            nstep = f_N(j, log_k) - f_N(j-1, log_k)
            nstep = nstep if nstep >= 1 else 1
            pe = 0
            eliminated_gen_idx = None

            for pos_idx, gen_idx in enumerate(k_list):
                if list_of_dist[gen_idx] in ...:
                    mmd_list = []
                    for s in range(nstep):
                        list_of_models[gen_idx].epochs = ... * (s + 1)
                        list_of_models[gen_idx].fit(...)
                        synthetic_samples = torch.tensor(list_of_models[gen_idx].sample(n_samples))
                        mmd_loss_fn = MMDLoss(kernel=RBF(gamma=lc_gamma))
                        reward = mmd_loss_fn(synthetic_samples, cluster)
                        mmd_list.append(reward)
                else:
                    raise ValueError(f"Unknown distribution name: {list_of_dist[gen_idx]}")
                reward_mean = -1 * sum(mmd_list) / len(mmd_list)
                print(mmd_list, reward_mean)
                if reward_mean <= pe:
                    pe = reward_mean
                    eliminated_gen_idx = pos_idx
                    el_gen = gen_idx
            k_list.pop(eliminated_gen_idx)
            print(f"Iteration {j}/{n_components - 1}, Eliminated Gen: {list_of_dist[el_gen]}")

        return list_of_dist[k_list[0]]
    
    else:
        raise ValueError(f"Unknown bandit type: {bandit_type}")

def pretrain(X, cluster_k, list_of_dist, n_samples=1000, max_iterations=100, discrete_columns=[], epsilon = 1, mh_gamma=1.0, unit_time_dict = dict()):
    assert len(X) == len(list_of_dist)
    n_components = len(X)
    list_of_models = []
    for i in range(n_components):
        iteration_size = int(max_iterations * unit_time_dict[list_of_dist[i]] / cluster_k)
        if list_of_dist[i] == 'ARF':
            list_of_models.append(arf.arf(epochs=iteration_size))
        elif list_of_dist[i] == ...:
            list_of_models.append(...)
        else:
            raise ValueError(f"Unknown distribution name: {list_of_dist[i]}")

    for j, cluster in enumerate(X):
        print(f"Pretrain Cluster {j + 1}, Gen: {list_of_dist[j]}")
        if list_of_dist[j] in ...:
            list_of_models[j].fit(...)
        else:
            raise ValueError(f"Unknown distribution name: {list_of_dist[j]}")

    return list_of_models

def generate_synthetic_data(selected_gen, X, n_samples, num_samples_comparison, weights, similarity_measure, burnin, nstep, list_of_models, max_iterations=1, batch_size=100, discrete_columns=[], mh_gamma=1.0, unit_time_dict = dict()):
    select_gen_multi = [unit_time_dict[gen] for gen in selected_gen]
    select_gen_multi_para = [int(time / min(select_gen_multi)) for time in select_gen_multi]

    mmd_values = []
    synthetic_samples = X
    n_components = len(list_of_models)

    plot_dir = ...
    csv_path = os.path.join(plot_dir, ...)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    open(csv_path, 'w').close()

    for iteration in range(max_iterations):
        synthetic_samples_list = []
        for j in range(len(list_of_models)):
            if list_of_models[j].type in ...:
                synthetic_samples_list.append(torch.tensor(list_of_models[j].sample(num_samples_comparison)))
            else:
                ...

        # Membership probility matrix computation
        membership_probabilities = torch.zeros((n_samples, n_components))
        for i in range(n_samples):
            for j in range(n_components):
                similarity = compute_similarity(X[i], synthetic_samples_list[j], similarity_measure)
                membership_probabilities[i, j] = similarity * weights[j]
        membership_probabilities = torch.clamp(membership_probabilities, min=1e-3)  # Avoid zero probabilities

        # Update parameters
        row_sums = membership_probabilities.sum(dim=1, keepdim=True)
        membership_probabilities = membership_probabilities / row_sums

        col_sums = membership_probabilities.sum(dim=0, keepdim=True)
        membership_probabilities_batch = membership_probabilities / col_sums
        for j in range(n_components):
            batch_indices = torch.multinomial(membership_probabilities_batch[:, j], n_samples, replacement=True)
            X_batch = X[batch_indices]

            iteration_size = ...
            
            if list_of_models[j].type in ...:
                list_of_models[j].epochs = iteration_size
                list_of_models[j].fit(X_batch.detach().numpy())
            else:
                ...
        
        # Calculate optim weights
        col_sums = membership_probabilities.sum(dim=0)
        weights = col_sums / n_samples
        optim_weights = col_sums / col_sums.sum()

        component_indices = torch.multinomial(optim_weights, n_samples, replacement=True)
        optim_num_list = [(component_indices == idx).nonzero(as_tuple=True)[0] for idx in range(len(list_of_models))]
        optim_num_list = [indices.size(dim=0) for indices in optim_num_list]
        optim_num_list = [int(num) if num >= 1 else 1 for num in optim_num_list]

        synthetic_samples = []
        for j in range(len(list_of_models)):
            if optim_num_list[j] == 0:
                print(f"Skipping model {list_of_models[j].type} due to zero samples.")
                continue
            elif list_of_models[j].type in ...:
                synthetic_samples.append(torch.tensor(list_of_models[j].sample(optim_num_list[j])))
            else:
                ...
        list_synthetic_samples = synthetic_samples.copy()
        synthetic_samples = torch.vstack(synthetic_samples)

        mmd_loss_fn = MMDLoss(kernel=RBF(gamma=mh_gamma))
        mmd = mmd_loss_fn(synthetic_samples, X)
        mmd_values.append(np.round(mmd.item(), 4))

        if iteration % 50 == 0:
            plt.scatter(X.detach().numpy()[:, 0], X.detach().numpy()[:, 1], color="blue", label="Train Data", alpha=0.5)
            for i in range(len(list_synthetic_samples)):
                plt.scatter(list_synthetic_samples[i].detach().numpy()[:, 0], list_synthetic_samples[i].detach().numpy()[:, 1], color=..., label=f"Generated Cluster {i + 1}", alpha=0.2)
            plt.legend()
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.title("Mixture Model Sample Data")
            plt.savefig(f'{plot_dir}/Original_and_Synthetic_Data_Ite{iteration + 1}.png')
            plt.clf()

        print(f"\rIteration {iteration + 1}/{max_iterations}: MMD = {mmd_values[-1].item()}, Optim_Weights = {optim_weights}", end='', flush=True)
        with open(csv_path, 'ab') as f:
            np.savetxt(f, [np.round(optim_weights.numpy(), 4)], delimiter=',')
        
    print()
    print(f'Weights: {optim_weights}')

    return {'samples': synthetic_samples, 'list_samples': list_synthetic_samples, 'weights': optim_weights, 'mmd_values': mmd_values}


# Exp Begin

# Example Dataset
htru2 = fetch_ucirepo(id=267)
dataset = htru2.data.features

# @misc{banknote_authentication_267,
#   author       = {Lohweg, Volker},
#   title        = {{Banknote Authentication}},
#   year         = {2012},
#   howpublished = {UCI Machine Learning Repository},
#   note         = {{DOI}: https://doi.org/10.24432/C55P57}
# }

# Convert the dataset to a PyTorch tensor
dataset = torch.tensor(dataset.values, dtype=torch.float32)
# Randomly select 1000 data points from the dataset
indices = torch.randperm(dataset.size(0))
dataset = dataset[indices]

n_samples = dataset.shape[0]
n_features = dataset.shape[1]
discrete_columns = [f'{i}' for i in range(n_features)]

pretrain_portion = 0.2
pretrain_dataset = dataset[:int(n_samples * pretrain_portion)]
train_dataset = dataset[int(n_samples * pretrain_portion):]

# Calculate the gamma for RBF with Median Heuristic
mh_gamma = rbf_median(dataset)

# Cluster Initiating
labels, cluster_k = lim_cluster_tabular_data(pretrain_dataset.detach().numpy(), 10)

clusters = [pretrain_dataset[labels == i] for i in range(cluster_k)]

eval_max_ite = ...
bandit_type = 'R-SR'
list_of_available_gen = ['ARF', ...]

unit_time_dict = ...
selected_gen = [bandit(unit_time_dict, cluster, list_of_available_gen, bandit_type=bandit_type, n_samples=100, n_features=n_features, 
                       budget=eval_max_ite, window_size=0.25, explore_para=0.1, discrete_columns=discrete_columns, epsilon = 1, mh_gamma=mh_gamma) for cluster in clusters]

# Budget Computation
budget = 10
pretrain_budget = 0.2
pretrain_ite, num_of_ite = ..., ...

# Step 1
train_list_of_models = pretrain(clusters, cluster_k, selected_gen, n_samples=n_samples, max_iterations=pretrain_ite, discrete_columns=discrete_columns, epsilon = 1, mh_gamma=mh_gamma, unit_time_dict = unit_time_dict)

train_weights = torch.tensor([len(cluster) / pretrain_dataset.shape[0] for cluster in clusters])

# Step 2

synthetic_data = generate_synthetic_data(X=train_dataset, n_samples=train_dataset.shape[0], num_samples_comparison=train_dataset.shape[0], 
                                         weights=train_weights, similarity_measure='new_es', burnin=10, nstep=..., list_of_models=train_list_of_models, 
                                         max_iterations=num_of_ite, batch_size=100, discrete_columns=discrete_columns, mh_gamma=mh_gamma, unit_time_dict = unit_time_dict)

evl_result = metrics_evaluation(synthetic_data['samples'], train_dataset)
print(evl_result)