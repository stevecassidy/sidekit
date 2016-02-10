# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2015 Anthony Larcher

:mod:`sv_utils` provides utilities to facilitate the work with SIDEKIT.
"""
import numpy as np
import scipy as sp
import pickle
import gzip
import os


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2016 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def save_svm(svmFileName, w, b):
    """Save SVM weights and bias in PICKLE format
    
    :param svmFileName: name of the file to write
    :param w: weight coefficients of the SVM to store
    :param b: biais of the SVM to store
    """
    if not os.path.exists(os.path.dirname(svmFileName)):
            os.makedirs(os.path.dirname(svmFileName))
    with gzip.open(svmFileName, "wb") as f:
            pickle.dump((w, b), f)


def read_svm(svmFileName):
    """Read SVM model in PICKLE format
    
    :param svmFileName: name of the file to read from
    
    :return: a tupple of weight and biais
    """
    with gzip.open(svmFileName, "rb") as f:
        (w, b) = pickle.load(f)
    return np.squeeze(w), b


def check_file_list(inputFileList, fileDir, fileExtension):
    """Check the existence of a list of files in a specific directory
    Return a new list with the existing segments and a list of indices 
    of those files in the original list. Return outputFileList and 
    idx such that inputFileList[idx] = outputFileList
    
    :param inputFileList: list of file names
    :param fileDir: directory where to search for the files
    :param fileExtension: extension of the files to search for
    
    :return: a list of existing files and the indices 
        of the existing files in the input list
    """
    existFiles = np.array([os.path.isfile(os.path.join(fileDir, f + fileExtension)) for f in inputFileList])
    outputFileList = inputFileList[existFiles, ]
    idx = np.argwhere(np.in1d(inputFileList, outputFileList))
    return outputFileList, idx.transpose()[0]


def initialize_iv_extraction_weight(ubm, T):
    """
    Estimate matrices W and T for approximation of the i-vectors
    For more information, refers to [Glembeck09]_

    :param ubm: Mixture object, Universal Background Model
    :param T: Raw TotalVariability matrix as a ndarray
    
    :return:    
      W: fix matrix pre-computed using the weights from the UBM and the 
          total variability matrix
      Tnorm: total variability matrix pre-normalized using the co-variance 
          of the UBM
    """
    # Normalize the total variability matrix by using UBM co-variance
    
    sqrt_invcov = np.sqrt(ubm.get_invcov_super_vector()[:, np.newaxis])
    Tnorm = T * sqrt_invcov
    
    # Split the Total Variability matrix into sub-matrices per distribution
    Tnorm_c = np.array_split(Tnorm, ubm.distrib_nb())
    
    # Compute fixed matrix W
    W = np.zeros((T.shape[1], T.shape[1]))
    for c in range(ubm.distrib_nb()):
        W = W + ubm.w[c] * np.dot(Tnorm_c[c].transpose(), Tnorm_c[c])

    return W, Tnorm


def initialize_iv_extraction_eigen_decomposition(ubm, T):
    """Estimate matrices Q, D_bar_c and Tnorm, for approximation 
    of the i-vectors.
    For more information, refers to [Glembeck09]_
    
    :param ubm: Mixture object, Universal Background Model
    :param T: Raw TotalVariability matrix
    
    :return:
      Q: Q matrix as described in [Glembeck11]
      D_bar_c: matrices as described in [Glembeck11]
      Tnorm: total variability matrix pre-normalized using the co-variance of the UBM
    """
    # Normalize the total variability matrix by using UBM co-variance
    sqrt_invcov = np.sqrt(ubm.get_invcov_super_vector()[:, np.newaxis])
    Tnorm = T * sqrt_invcov
    
    # Split the Total Variability matrix into sub-matrices per distribution
    Tnorm_c = np.array_split(Tnorm, ubm.distrib_nb())
    
    # Compute fixed matrix Q
    W = np.zeros((T.shape[1], T.shape[1]))
    for c in range(ubm.distrib_nb()):
        W = W + ubm.w[c] * np.dot(Tnorm_c[c].transpose(), Tnorm_c[c])
    
    eigenValues, Q = sp.linalg.eig(W)
    
    # Compute D_bar_c matrix which is the diagonal approximation of Tc' * Tc
    D_bar_c = np.zeros((ubm.distrib_nb(), T.shape[1]))
    for c in range(ubm.distrib_nb()):
        D_bar_c[c, :] = np.diag(reduce(np.dot, [Q.transpose(), Tnorm_c[c].transpose(), Tnorm_c[c], Q]))
    return Q, D_bar_c, Tnorm


def initialize_iv_extraction_fse(ubm, T):
    """Estimate matrices for approximation of the i-vectors.
    For more information, refers to [Cumani13]_
    
    :param ubm: Mixture object, Universal Background Model
    :param T: Raw TotalVariability matrix
    
    :return:
      Q: Q matrix as described in [Glembeck11]
      D_bar_c: matrices as described in [Glembeck11]
      Tnorm: total variability matrix pre-normalized using the co-variance of the UBM
    """
    # TODO: complete documentation and write the code

    # % Initialize the process
    # %init = 1;
    #
    #
    #   Extract i-vectors by using different methods
    #
    # %rank_T      = 10;
    # %featureSize = 50;
    # %distribNb   = 32;
    # %dictSize    = 5;
    # %dictSizePerDis=rank_T;  % a modifier par la suite
    #
    # %ubm_file    = 'gmm/world32.gmm';
    # %t_file      = 'mat/TV_32.matx';
    #
    #
    #
    #   Load data
    #
    #
    # % Load UBM for weight parameters that are used in the optimization
    # % function
    # %UBM = ALize_LoadModel(ubm_file);
    #
    # % Load meand from Minimum Divergence re-estimation
    # %sv_mindiv = ALize_LoadVect(minDiv_file)';
    #
    # % Load T matrix
    # %T = ALize_LoadMatrix(t_file)';
    #
    #
    # function [O,PI,Q] = factorized_subspace_estimation(UBM,T,dictSize,outIterNb,inIterNb)
    #
    #    rank_T      = size(T,2);
    #    distribNb   = size(UBM.W,2);
    #    featureSize = size(UBM.mu,1);
    #
    #    % Normalize matrix T
    #    sv_invcov = reshape(UBM.invcov,[],1);
    #    T_norm = T .* repmat(sqrt(sv_invcov),1,rank_T);
    #
    #     Tc{1,distribNb} = [];
    #     for ii=0:distribNb-1
    #         % Split the matrix in sub-matrices
    #         Tc{ii+1} = T_norm((ii*featureSize)+1:(ii+1)*featureSize,:);
    #     end
    #
    #     % Initialize O and Qc by Singular Value Decomposition
    #     init_FSE.Qc{distribNb} 	= [];
    #     init_FSE.O{distribNb}   = [];
    #     PI{distribNb}           = [];
    #
    #     for cc=1:distribNb
    #         init_FSE.Qc{cc} 	= zeros(featureSize,featureSize);
    #         init_FSE.O{cc}   = zeros(featureSize,featureSize);
    #         PI{cc}  = sparse(zeros(featureSize,dictSize*distribNb));      % a remplacer par une matrice sparse
    #     end
    #
    #     % For each distribution
    #     for cc=1:distribNb
    #         fprintf('Initilize matrice for distribution %d / %d\n',cc,distribNb);
    #         % Initialized O with Singular vectors from SVD
    #         [init_FSE.O{cc},~,V] = svd(UBM.W(1,cc)*Tc{cc});
    #         init_FSE.Qc{cc} = V'; 
    #     end
    #
    #
    #    % Concatenate Qc to create the matrix Q: A MODIFIER POUR DISSOCIER
    #    % dictSize DU NOMBRE DE DISTRIBUTIONS
    #    Q = [];
    #    for cc=1:distribNb
    #        Q = [Q;init_FSE.Qc{cc}(1:dictSize,:)];
    #    end
    #    O = init_FSE.O;
    #    clear 'init_FSE'
    #
    #
    #    % OUTER iteration process : update Q iteratively
    #    for it = 1:outIterNb
    #
    #        fprintf('Start iteration %d / %d for Q re-estimation\n',it,5);
    #
    #        % INNER iteration process: update PI and O iteratively
    #        for pioIT = 1:inIterNb
    #            fprintf('   Start iteration %d / %d for PI and O re-estimation\n',pioIT,10);
    #
    #            % Update PI
    #            %Compute diagonal terms of QQ'
    # %            diagQ = diag(Q*Q');
    #
    #            for cc=1:distribNb
    #
    #                % Compute optimal k and optimal v 
    #                % for each line f of PI{cc}
    #                A = O{cc}'*Tc{cc}*Q';
    #                f = 1;
    #                while (f < size(A,1)+1)
    #
    #                    if(f == 1)
    #                        A = O{cc}'*Tc{cc}*Q';       % equation (26)
    #                        PI{cc} = sparse(zeros(featureSize,dictSize*distribNb));
    #                    end
    #
    #                    % Find the optimal index k
    #                    [~,k] = max(A(f,:).^2);
    #                    k_opt = k;
    #
    #                    % Find the optimal value v
    #                    v_opt = A(f,k_opt);
    #
    #                    % Update the line of PI{cc} with the value v_opt in the
    #                    % k_opt-th column
    #                    PI{cc}(f,k_opt)     = v_opt;
    #
    #                    % if the column already has a non-zero element,
    #                    % update O and PI
    #                    I = find(PI{cc}(:,k_opt)~=0);
    #                    if size(I,1)>1
    #                        % get indices of the two lines of PI{cc} which
    #                        % have a non-zero element on the same column
    #                        a = I(1);
    #                        b = I(2);
    #
    #                        % Replace column O{cc}(:,a) and O{cc}(:,b)
    #                        Oa = (PI{cc}(a,k_opt)*O{cc}(:,a)+PI{cc}(b,k_opt)*O{cc}(:,b))/(sqrt(PI{cc}(a,k_opt)^2+PI{cc}(b,k_opt)^2));
    #                        Ob = (PI{cc}(a,k_opt)*O{cc}(:,b)-PI{cc}(b,k_opt)*O{cc}(:,a))/(sqrt(PI{cc}(a,k_opt)^2+PI{cc}(b,k_opt)^2));
    #                        O{cc}(:,a) = Oa;
    #                        O{cc}(:,b) = Ob;
    #
    #                        PI{cc}(a,k_opt) = sqrt(PI{cc}(a,k_opt)^2+PI{cc}(b,k_opt)^2);
    #                        PI{cc}(b,k_opt) = 0;
    #
    #                        f = 0;
    #                    end
    #                    f =f +1;
    #                end
    #            end
    #
    #            obj = computeObjFunc(UBM.W,Tc,O,PI,Q);
    #            fprintf('Objective Function after estimation of PI = %2.10f\n',obj);                
    #
    #            % Update O
    #            for cc=1:distribNb
    #
    #                % Compute 
    #                Z = PI{cc}*Q*Tc{cc}';
    #
    #                % Compute Singular value decomposition of Z
    #                [Uz,~,Vz] = svd(Z);
    #
    #                % Compute the new O{cc}
    #                O{cc} = Vz*Uz';
    #            end
    #            obj = computeObjFunc(UBM.W,Tc,O,PI,Q);
    #            fprintf('Objective Function after estimation of O = %2.10f\n',obj);
    #        end % END OF INNER ITERATION PROCESS
    #
    #
    #        % Update Q
    #        D = sparse(zeros(size(PI{cc},2),size(PI{cc},2)));
    #        E = zeros(size(PI{cc},2),rank_T);
    #        for cc=1:distribNb
    #            % Accumulate D
    #            D = D + UBM.W(1,cc) * PI{cc}'*PI{cc};
    #
    #            % Accumulate the second term
    #            E = E + UBM.W(1,cc) * PI{cc}'*O{cc}'*Tc{cc};
    #        end
    #        Q = D\E;
    #
    #        % Normalize rows of Q and update PI accordingly
    #
    #        % Compute norm of each row
    #        c1 = bsxfun(@times,Q,Q);
    #        c2 = sum(c1,2);
    #        c3 = sqrt(c2);
    #        Q = bsxfun(@rdivide,Q,c3);
    #
    #        % Update PI accordingly
    #        PI = cellfun(@(x)x*sparse(diag(c3)), PI, 'uni',false);
    #
    #        obj = computeObjFunc(UBM.W,Tc,O,PI,Q);
    #        fprintf('Objective Function after re-estimation of Q = %2.10f\n',obj);
    #    end
    # end
    pass


