
function [laKMM, laMM, BiGraph, A, OBJ, Ah, laKMMh] = KMM_mmconv(X, c, m, k)
% [laKMM, laMM, BiGraph, Anc, ~, ~, ~]= KMM(X', c, m,k) : K-Multiple-Means
% Input:
%       - X: the data matrix of size nFea x nSmp, where each column is a sample
%               point
%       - c: the number of clusters
%       - m: the number of multiple means(MM)
%       - k: the number of neighbor points
% Output:
%       - laKMM: the cluster assignment for each point
%       - laMM: the sub-cluster assignment for each point
%       - BiGraph: the matrix of size nSmp x nMM
%       - A: the multiple means matrix of size nFea x nMM
%       - Ah: the history of multiple means
%       - laKMMh: the history of cluster assignment for each point
% Requre:
%       CSBG.m
% 		meanInd.m
% 		ConstructA_NP.m
% 		EProjSimplex_new.m
% 		svd2uv.m
% 		struG2la.m
%       eig1.m
%       gen_nn_distanceA.m
% Usage:
%       % X: d*n
%       [laKMM, laMM, AnchorGraph, Anchors, ~, ~, ~]= KMM(X', c, m,k) ;
% Reference:
%
%	Feiping Nie, Cheng-Long Wang, Xuelong Li, "K-Multiple-Means: A Multiple-Means 
%   Clustering Method with Specified K Clusters," In The 25th ACM SIGKDD Conference
%   on Knowledge Discovery and Data Mining (KDD ’19), August 4–8, 2019, Anchorage, AK, USA.
%
%   version 1.0 --May./2019 
%
%   Written by Cheng-Long Wang (ch.l.w.reason AT gmail.com)
if nargin < 4
    if m<6
        k=c-1;
    else
        k=5;
    end      
end
Ah=[];
laKMMh=[];
Iter=15;
OBJ=[];
n=size(X,2);
method=1; % method for initial seeds, 1:kmeans; 0:random 
opt_conv=1; % option for convergence, 1:sub prototypes; 0:partiton of subclusters 
% StartIndZ: before MM update
if method ==0
    StartIndZ=randsrc(n,1,1:m);
else
    StartIndZ=kmeans(X',m);
end
BiGraph = ones(n,m);
A = meanInd(X, StartIndZ,m,BiGraph);
[laKMM, laMM, BiGraph, isCov, obj, ~] = CSBG(X, c, A, k);
% fprintf('time:%d,obj:%d\n',ti,obj)
iter=1;
while(iter<Iter)
    iter = iter +1;
    if isCov
        laKMMh=[laKMMh laKMM];
        Ah=[Ah A];
        OBJ=[OBJ obj];
        if opt_conv==1
            StartIndZ=laMM; 
            A_old = A;
            A = meanInd(X, StartIndZ, m, BiGraph);
            Dis = sqdist(A_old,A); % O(ndm)
            distXt = Dis;
            di = min(distXt, [], 2);  
            if norm(di)<1e-4
                fprintf('means converge\n')
                return;
            end
        else            
            if (all(StartIndZ==laMM))
                fprintf('partition converge\n')
                return;
            else
                StartIndZ=laMM; 
            end                   
        end
        [laKMM, laMM, BiGraph, isCov, obj, ~] = CSBG(X, c, A, k);   
    else
        if method ==0
            StartIndZ=randsrc(n,1,1:m);
        else
            StartIndZ=kmeans(X',m);
        end
        BiGraph = ones(n,m);
        A = meanInd(X, StartIndZ,m,BiGraph);
        [laKMM, laMM, BiGraph, isCov, obj, ~] = CSBG(X, c, A, k);
    end
fprintf('loop:%d\n',iter)
end
