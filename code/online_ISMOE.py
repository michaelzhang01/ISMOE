# -*- coding: utf-8 -*-
"""
Created on Sat Mar 2 11:13:22 2019
Online Minibatch Importance Sampled Mixture of Experts (ISMOE) code

@author: Michael Zhang
"""
# Python 2 to 3 conversion
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from past.builtins import xrange
from past.builtins import map
import numpy as np
import GPy
import time
import pdb
from mpi4py import MPI
from scipy import stats
from scipy.misc import logsumexp
from scipy.sparse import lil_matrix
from itertools import product
from scipy.optimize import minimize, brent
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import KFold
from GPy.util.univariate_Gaussian import std_norm_cdf
from GPy.util.linalg import jitchol, dtrtrs
from mvn_t import log_mvn_t
from scipy.io import loadmat
from math import *
import matplotlib.pyplot as plt

def _unscaled_dist(X, X2=None):
    if X2 is None:
        Xsq = np.sum(np.square(X),1)
        r2 = -2.*GPy.util.linalg.tdot(X) + (Xsq[:,None] + Xsq[None,:])
        GPy.util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)
    else:
        X1sq = np.sum(np.square(X),1)
        X2sq = np.sum(np.square(X2),1)
        r2 = -2.*np.dot(X, X2.T) + X1sq[:,None] + X2sq[None,:]
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

def pad_kernel_matrix(K, X, X_star):
    assert(K.shape[0] == K.shape[1])
    N, D = X.shape
    assert(K.shape[0] == N)
    assert(X.shape[1] == X_star.shape[1])
    N_star,_ = X_star.shape
    for X_star_i in X_star:
        K = np.pad(K,[(0,1),(0,1)],'constant',constant_values=0)
        x_star_i_k = _unscaled_dist(X,X_star_i[np.newaxis,:]).flatten()
        K[-1,:-1] = x_star_i_k
        K[:-1,-1] = x_star_i_k
        X = np.vstack((X,X_star_i))
    return(K)
#        pad_K[-1:]

class ISMOE(object):
    def __init__(self,X, Y, X_star, Y_star, K=50, alpha=1.,J=2,
                 N_minibatch = 1000, full_cov=True, mb_upweight=True,
                 prior_obs=1.):
        """
        ISMOE code for classification and regression.
        Parameters:
            X: N x D Numpy array of training inputs.
            Y: N x 1 Numpy array of training outputs
            X_star: N_star x D Numpy array of test inputs
            Y_star N_star x 1 Numpy array of test outputs
            K: Integer, number of cluster components per importance sample
            alpha: Float, concentration parameter of Dirichlet mixture model
            J: Integer, number of importance samples
            classification: Bool, True to run classification and False
                            for regression
            partition: String, partition types. Availble options are "gmm",
                        "kmeans", "random", and "vi"
            IS: Bool, weighting type. True for importance weights, False for
                uniform weights.
            Stationary: Bool, True for fitting a stationary kernel, False for
                        Non-stationary
            N_minibatch: Integer, Size of stochastic approximation, must be
                         less than N. Ignored for classification.
            full_cov: Bool, True if you want full covariance matrix returned in
                      predictions. False if you want just the diagonal.
            mb_upweight: Bool, True if you want to upweight likelihood for
                         stochastic approximation. Ignored for classifcation.
        """
        self.start_time = time.time()
        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.total_J = int(J)
        self.full_cov = bool(full_cov)
        self.proposal_sizes = [j.size for j in np.array_split(xrange(self.total_J), self.P)]
        assert(len(self.proposal_sizes) == self.P)
        assert(sum(self.proposal_sizes) == self.total_J)
        self.prior_obs = prior_obs
        assert(self.prior_obs >0.)
        self.J = self.proposal_sizes[self.rank]
        self.K = np.zeros(self.J).astype(int)
        self.alpha=alpha
        self.X = X
        self.Y = np.array(Y).reshape(-1,1)
        (self.N, self.D) = self.X.shape
        self.M = 3 # num of hyperparameters, fixed to RBF kernel for now
        self.N_minibatch = N_minibatch
        self.init_hyp = np.empty((self.J,self.M))
        assert(self.N_minibatch <= self.N)
        if mb_upweight:
            self.mb_weight = float(self.N) / float(self.N_minibatch)
        else:
            self.mb_weight = 1.
        assert(self.N == self.Y.size)
        self.U = [np.random.choice(self.N,size=self.N_minibatch,replace=False)for j in xrange(self.J)]
        self.X_mean = self.X.mean(axis=0)
        self.X_sd = np.sqrt(self.X.var(axis=0))
        self.W = np.zeros(self.J)
        # everything J should be a list, K should be a dictionary.
        self.marginal_LL_k = [None]*self.J
        Z_map = map(self.partition_data,range(self.J))
        self.Z_proposal = [Z for Z,Z_count in Z_map]
        self.Z_count = [Z_count for Z,Z_count in Z_map]
        self.kernels = [None]*self.J # np.zeros().astype(object) #{j:None for j in xrange(self.J)}
        self.models = map(self.model_init,xrange(self.J))
        self.marginal_LL = np.copy(self.W)
        self.comm.barrier()

    def partition_data(self,j):
        dp = BayesianGaussianMixture(n_components = int(self.alpha*np.log(self.N)) ,weight_concentration_prior = self.alpha, init_params='kmeans',weight_concentration_prior_type='dirichlet_process')
        Z = dp.fit_predict(self.X[self.U[j]])
        le = LE()
        Z = le.fit_transform(Z)
        Z_count = np.bincount(Z)
        assert(Z.max()+1 == Z_count.size)
        self.K[j] = int(Z_count.size)
        self.marginal_LL_k[j] = {k:0 for k in range(int(self.K[j])) }
        return(Z,Z_count)

    def posterior_mvn_t(self,X_k,X_star_i):
        N_k = X_k.shape[0]
        assert(N_k >= 0)
        if N_k > 0:
#            data_k = X[np.where(Z==k)]  #self.X[np.where(self.Z == k)]
            X_bar = X_k.mean(axis=0)
            diff = X_k - X_bar
        else:
            X_bar = np.zeros(self.D)
            diff = np.zeros(self.D)
        mu_posterior = (self.prior_obs*self.X_mean + N_k*X_bar)/(N_k + self.prior_obs)
        SSE = np.dot(diff.T, diff)
        prior_diff = X_bar - self.X_mean
        SSE_prior = np.outer(prior_diff.T, prior_diff)
        nu_posterior = self.D + 2 + N_k
        lambda_posterior = self.prior_obs + N_k
        psi_posterior = (self.X_sd**-2)*np.eye(self.D) + SSE + (self.prior_obs * N_k)/(self.prior_obs + N_k) * SSE_prior
        psi_posterior *= (lambda_posterior + 1.)/(lambda_posterior*(nu_posterior - self.D + 1.))
        df_posterior = (nu_posterior - self.D + 1.)
        return(log_mvn_t(X_star_i, mu_posterior, psi_posterior, df_posterior))


    def crp_predict(self,X_star,X,Z,Z_count,restrict=True):
        max_k = max(Z)
        assert(X_star.shape[1] == self.D)
        N_star, _ = X_star.shape
        for i in range(N_star):
            if restrict:
                log_prob = np.log(Z_count)
            else:
                log_prob = np.log(np.concatenate((Z_count,[self.alpha])))
            log_prob += [self.posterior_mvn_t(X[np.where(Z==k)],X_star[i]) for k in range(log_prob.size)]
            Z_i = log_prob.argmax()
            Z = np.append(Z,Z_i)
            if restrict:
                Z_count[Z_i] += 1
            else:
                if Z_i > max_k:
                    Z_count= np.append(Z_count,1)
                    max_k += 1
                else:
                    Z_count[Z_i] += 1
            X = np.vstack((X,X_star[i]))
        return(Z,Z_count)

    def model_init(self,j):
        init_n = 10 # initialize hyperparameters with a tiny subset of minibatch
        unique_Z_j = np.unique(self.Z_proposal[j])
        gp_model = {}
        self.kernels[j] = {k:_unscaled_dist(self.X[self.U[j]][self.Z_proposal[j]==k]) for k in unique_Z_j}
        if self.N_minibatch > init_n:
            choose_init =  np.random.choice(self.N_minibatch,size=init_n,replace=False)
            m0_flag = True
            m0 = GPy.models.GPRegression(self.X[self.U[j]][choose_init], self.Y[self.U[j]][choose_init].reshape(-1,1), kernel = GPy.kern.RBF(self.D))
            while m0_flag: # sometimes initial choice returns cholesky error
                try:
                    m0.optimize()
                    m0_flag=False
                except:
                    choose_init =  np.random.choice(self.N_minibatch,size=init_n,replace=False)
                    m0 = GPy.models.GPRegression(self.X[self.U[j]][choose_init], self.Y[self.U[j]][choose_init].reshape(-1,1), kernel = GPy.kern.RBF(self.D))

            self.init_hyp[j] = np.log(m0.param_array)
            del m0
        else:
            self.init_hyp[j] = np.ones(self.M)

#        unique_Z_j = np.unique(self.Z_proposal[j])
        for k in unique_Z_j:
            K_mask = (self.Z_proposal[j]==k)
            Y_k = np.copy(self.Y[self.U[j]][K_mask]).reshape(-1,1)
            X_k = np.copy(self.X[self.U[j]][K_mask])
            min_f=minimize(self.neg_log_marg_LL_ns,self.init_hyp[j], args=(self.kernels[j][k],Y_k))
            hyp = np.exp(min_f.x)
            gp_model[k] = GPy.models.GPRegression(X_k,Y_k, kernel = GPy.kern.RBF(self.D, lengthscale=hyp[0],variance=hyp[1]), noise_var=hyp[2])
            if self.mb_weight == 1:
                marg_LL_k = gp_model[k].log_likelihood()
            else:
                marg_LL_k =  -1.*min_f.fun
            self.W[j] += marg_LL_k
            self.marginal_LL_k[j][k] =marg_LL_k
        return(gp_model)

    def generate_predictions(self,j,X_star,Y_star = None):
        assert(X_star.shape[1] == self.D)
        N_star,_ = X_star.shape
        temp_Z,temp_Z_count = self.crp_predict(X_star,self.X[self.U[j]],self.Z_proposal[j],self.Z_count[j])
        Z_star = temp_Z[self.N:]
        Y_star_predict = np.empty(N_star)
        if Y_star:
            Y_star_LL = 0
        else:
            Y_star_LL = None

        if self.full_cov:
            sparse_cov = lil_matrix((N_star,N_star))
        else:
            sparse_cov = np.zeros(N_star)

        for k in np.unique(Z_star):
            K_star_mask = (Z_star == k)
            assert(K_star_mask.size == N_star)
            X_star_k = X_star[K_star_mask]
            pred_GP_mean, pred_GP_cov = self.models[j][k].predict(X_star_k,full_cov=self.full_cov)
            if self.full_cov:
                for idx,x_k in enumerate(np.where(Z_star[j]==k)[0]):
                    sparse_cov[x_k,K_star_mask] += pred_GP_cov[idx,:].reshape(1,-1)
                    sparse_cov[K_star_mask,x_k] += pred_GP_cov[idx,:].reshape(-1,1)
            else:
                sparse_cov[K_star_mask] += pred_GP_cov.flatten()
            if Y_star:
                if self.full_cov:
                    Y_star_LL += stats.norm.logpdf(self.Y_star.flatten()[K_star_mask],self.Y_star_predict[j][K_star_mask].flatten(), np.sqrt(np.diag(self.Y_star_cov[j].toarray()))[K_star_mask].flatten()).sum()
                else:
                    Y_star_LL += stats.norm.logpdf(self.Y_star.flatten()[K_star_mask],self.Y_star_predict[j][K_star_mask].flatten(), np.sqrt(self.Y_star_cov[j][K_star_mask].toarray()).flatten()).sum()
            Y_star_predict[K_star_mask] = pred_GP_mean.flatten()

        if Y_star:
            return(Y_star_predict, sparse_cov, Y_star_LL)
        else:
            return(Y_star_predict, sparse_cov)

    def update_weight(self,j, X_star,Y_star, N_star_minibatch):
        N_star = Y_star.shape[0]
        if self.mb_weight == 1:
            U_star = np.arange(N_star).astype(int)
        else:
            U_star = np.sort(np.random.choice(N_star,replace=False,size=N_star_minibatch))
        self.Z_proposal[j],self.Z_count[j] = self.crp_predict(X_star[U_star],self.X[self.U[j]],self.Z_proposal[j],self.Z_count[j],restrict=False)
        Z_star_proposal = self.Z_proposal[j][self.N:]
        Z_N_proposal = self.Z_proposal[j][:self.N]
        Z_star_unique = np.unique(Z_star_proposal)
        for k in Z_star_unique:
            X_star_k = X_star[U_star][Z_star_proposal==k]
            Y_star_k = Y_star[U_star][Z_star_proposal==k]
            if k < self.K[j]:# join existing model
                X_k = self.X[self.U[j]][Z_N_proposal==k]
                Y_k = self.Y[self.U[j]][Z_N_proposal==k]
                new_Y = np.vstack((Y_k,Y_star_k))
                self.kernels[j][k] = pad_kernel_matrix(self.kernels[j][k], X_k,X_star_k)
                self.models[j][k].set_XY( np.vstack((X_k,X_star_k)),  new_Y)
                marg_LL_k = -1.*self.neg_log_marg_LL_ns( np.log(self.models[j][k].param_array), self.kernels[j][k],new_Y)
            else:# fit new model
                self.kernels[j][k] = _unscaled_dist(X_star_k)
                min_f=minimize(self.neg_log_marg_LL_ns,self.init_hyp[j], args=(self.kernels[j],Y_star_k,))
                hyp = np.exp(min_f.x)
                self.models[j][k] = GPy.models.GPRegression(X_star_k,Y_star_k, kernel = GPy.kern.RBF(self.D, lengthscale=hyp[0],variance=hyp[1]), noise_var=hyp[2])
                marg_LL_k = -1.*min_f.fun
                self.K[j] += 1
            assert(self.kernels[j][k].shape[0] == new_Y.size)
            self.marginal_LL_k[j][k] = marg_LL_k
        self.U[j] = np.append(self.U[j],U_star+self.N)
        self.W[j] -= self.marginal_LL[j] # old marg LL
        self.marginal_LL[j] = sum(self.marginal_LL_k[j][k] for k in xrange(self.K[j])) # new marginal likelihood calculation
        self.W[j] += self.marginal_LL[j] # new marg LL

    def new_data_update(self,X_star,Y_star):
        assert(X_star.shape[1] == self.D)
        assert(Y_star.shape[0]==X_star.shape[0])
        N_star = Y_star.shape[0]
        if self.mb_weight == 1:
            N_star_minibatch = N_star
        else:
            N_star_minibatch = int( ( (self.N + N_star) / self.mb_weight) +  self.N_minibatch)
        [self.update_weight(j,X_star,Y_star,N_star_minibatch) for j in range(self.J)]
        self.normalize_weights()
        self.X = np.vstack((self.X,X_star))
        self.Y = np.vstack((self.Y,Y_star))
        self.N += N_star
        self.N_minibatch += N_star_minibatch

    def resample_particles(self):
        # gather everything indexed by j
        self.comm.barrier()
        gather_weights = self.comm.gather((self.W))
        gather_U = self.comm.gather(self.U)
        gather_models = self.comm.gather(self.models)
        gather_marginal_LL_k = self.comm.gather(self.marginal_LL_k)
        gather_K = self.comm.gather(self.K)
        gather_Z = self.comm.gather(self.Z_proposal)
        gather_Z_count = self.comm.gather(self.Z_count)
        gather_kernels = self.comm.gather(self.kernels)
        gather_init_hyp = self.comm.gather(self.init_hyp)
        if self.rank == 0:
            gather_weights = np.hstack(gather_weights)
            exp_gather_weights = gather_weights- logsumexp(gather_weights)
            exp_gather_weights = np.exp(gather_weights)
            resample_J = np.random.multinomial(1, exp_gather_weights,size=self.total_J).argmax(axis=1)
            gather_weights = gather_weights[resample_J]
            gather_weights -= logsumexp(gather_weights)
            gather_weights = np.array_split(gather_weights,self.P)
            gather_U = np.array_split(np.vstack(gather_U)[resample_J],self.P)
            gather_models = np.array_split(np.hstack(gather_models)[resample_J],self.P)
            gather_marginal_LL_k = np.array_split(np.hstack(gather_marginal_LL_k)[resample_J],self.P)
            gather_K = np.array_split(np.hstack(gather_K)[resample_J],self.P)
            gather_Z = np.array_split(np.vstack(gather_Z)[resample_J],self.P)
            gather_Z_count = np.array_split(np.vstack(gather_Z_count)[resample_J],self.P)
            gather_kernels = np.array_split(np.hstack(gather_kernels)[resample_J],self.P)
            gather_init_hyp = np.array_split(np.vstack(gather_init_hyp)[resample_J],self.P)
        else:
            exp_gather_weights= None
            resample_J = None
            gather_weights = None
            gather_U = None
            gather_models = None
            gather_marginal_LL_k = None
            gather_K = None
            gather_Z = None
            gather_Z_count = None
            gather_kernels = None
            gather_init_hyp = None
        self.W = self.comm.scatter(gather_weights)
        self.U = self.comm.scatter(gather_U)
        self.models = self.comm.scatter(gather_models)
        self.K = self.comm.scatter(gather_K)
        self.marginal_LL_k = self.comm.scatter(gather_marginal_LL_k)
        self.marginal_LL = np.zeros(self.J)
        for j in range(self.J):
            for k in range(self.K[j]):
                self.marginal_LL[j] += self.marginal_LL_k[j][k]
        self.Z_proposal = self.comm.scatter(gather_Z)
        self.Z_count = self.comm.scatter(gather_Z_count)
        self.kernels = self.comm.scatter(gather_kernels)
        self.init_hyp = self.comm.scatter(gather_init_hyp)
#        pdb.set_trace()

    def neg_log_marg_LL_ns(self,hyp, norm_k, Y_k): # hyp is [log lengthscale, log amplitude, log gaussian noise]
        hyp = np.exp(hyp)
        if np.any(np.isinf(hyp)):
            return(np.inf)
        else:
            N_k = Y_k.size
            if N_k > 1:
                kernel_k = hyp[1]*np.exp(  -.5 * (norm_k/ hyp[0] )**2)
                kernel_k += (hyp[2]/(self.mb_weight)  + 1e-6)*np.eye(N_k)
                try:
                    Wi, LW, LWi, W_logdet = GPy.util.linalg.pdinv(kernel_k)
                    alpha, _ = GPy.util.linalg.dpotrs(LW, Y_k, lower=1)
                    LL =  0.5*(-N_k*self.mb_weight * np.log(2.*np.pi) -  W_logdet - np.sum(alpha * Y_k))
                except:
                    return(np.inf)
            else:
                kernel_k = (hyp[2]/(self.mb_weight)  + hyp[1])

                LL = 0.5*(-N_k*self.mb_weight*  np.log(2.*np.pi) - np.log(kernel_k) - (Y_k[0]**2 / kernel_k))
            return(-1.*LL)

    def normalize_weights(self):
        self.comm.barrier()
        self.W[np.where(np.isnan(self.W))] = -np.inf
        gather_weights = self.comm.gather((self.W))
        if self.rank ==0:
            gather_weights = np.hstack(gather_weights)
            gather_weights = np.array_split(gather_weights-logsumexp(gather_weights), self.P)
        else:
            gather_weights = None
        self.W = self.comm.scatter(gather_weights)


    def prediction_combine(self,X_star,Y_star=None):
        self.normalize_weights()
        pred_output = np.array( [self.generate_predictions(j,X_star,Y_star) for j in range(self.J)] )
        assert(pred_output.shape[0] == self.J)
        if self.full_cov:
            pred_output = np.sum(np.exp(np.tile(self.W, (2,1)).T)*pred_output,axis=0)
        else:
            pred_output = np.sum(np.exp(np.tile(self.W, (X_star.shape[0],2,1)).T)*pred_output,axis=0)
        self.comm.barrier()
        reduce_pred_output = self.comm.reduce(pred_output)
        return(reduce_pred_output)

    def hyperparameter_update(self,j): # perhaps eventually upgrade to HMC?
        unique_Z_j = np.unique(self.Z_proposal[j])
        marg_LL = 0
        for k in unique_Z_j:
            K_mask = (self.Z_proposal[j]==k)
            Y_k = np.copy(self.Y[self.U[j]][K_mask]).reshape(-1,1)
            try:
                min_f=minimize(self.neg_log_marg_LL_ns, np.log(self.models[j][k].param_array), args=(self.kernels[j][k],Y_k))
            except:
                pdb.set_trace()
            if self.mb_weight == 1:
                marg_LL_k = self.models[j][k].log_likelihood()
            else:
                marg_LL_k =  -1.*min_f.fun
            marg_LL += marg_LL_k
            self.marginal_LL_k[j][k] =marg_LL_k
        self.W[j] =marg_LL

    def resample_hyperparameters(self):
        [self.hyperparameter_update(j) for j in range(self.J)]
        self.normalize_weights()

if __name__ == '__main__':

#    K_range= range(10,110,20)
#    N= 1000
    J= 4
    data=loadmat('../data/ns_data5.mat')
    X = data['X']
    X_star = data['X_star']
    Y = data['Y']
    Y_star = data['Y_star']
    full_X = np.vstack((X,X_star))
    X_sort = np.argsort(full_X[:,0])
    full_X = full_X[X_sort,:]
    full_Y = np.vstack((Y,Y_star))
    full_Y = full_Y[X_sort,:]
    X,X_star = full_X[:750],full_X[750:800]
    Y,Y_star = full_Y[:750],full_Y[750:800]
    N,D = X.shape

    K=10
#    X_full = np.memmap("gmm_X_mm_full",mode='r',dtype='float32',shape=(12000,100))
#    Y_full = np.memmap("gmm_Y_mm_full",mode='r',dtype='float32',shape=(12000,1))
    igps = ISMOE(X = X, Y=Y, X_star= X_star, Y_star = Y_star,K=K, J=J,
                IS=True,classification=False,N_minibatch = N,
                partition="gmm", stationary=True, mb_upweight=True,
                full_cov=False)
    Y_pred = igps.prediction_combine(X_star)
    igps.new_data_update(X_star,Y_star)
    igps.resample_particles()
    igps.resample_hyperparameters()
#            break
#            igps.prediction_combine()

#    J= 128
#    dat = loadmat("../data/skin.mat")
#    X_full = dat['X']
#    Y_full = dat['Y'].flatten()
#    print("ISMOE Classification")
#    for K in K_range:
#        kf=KFold(n_splits=5, random_state=0,shuffle=True)
#        for train,test in kf.split(X_full,Y_full):
#            X,Y = X_full[train], Y_full[train]
#            X_star,Y_star = X_full[test], Y_full[test]
#            igps = ISMOE(X = X, Y=Y, X_star= X_star, Y_star = Y_star,K=K, J=J,
#                                IS=True,classification=True,N_minibatch = N,
#                                partition="gmm", stationary=True, mb_upweight=True,
#                                full_cov=False)
#            igps.prediction_combine()

