
function [laKMM, laMM, BiGraph, A, OBJ, Ah, laKMMh] = KMM(X, c, m, k)
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
%       - laKMMh: the history of cluster assignment for each point
% Requre:
%       CSBG.m
% 		meanInd.m
% 		ConstructA_NP.m
% 		EProjSimplex_new.m
% 		svd2uv.m
% 		struG2la.m
%       eig1.m
% Usage:
%       % X: d*n
%       [laKMM, laMM, AnchorGraph, Anchors, ~, ~, ~]= KMM(X', c, m,k) ;
% Reference:
%
%	Feiping Nie, Cheng-Long Wang, Xuelong Li, "K-Multiple-Means: A Multiple-Means 
%   Clustering Method with Specified K Clusters," In The 25th ACM SIGKDD Conference
%   on Knowledge Discovery and Data Mining (KDD ¡¯19), August 4¨C8, 2019, Anchorage, AK, USA.
%
%   version 1.0 --May./2019 
%
%   Written by Cheng-Long Wang (ch.l.w.reason AT gmail.com)

Ah=[];
laKMMh=[];
Iter=15;
OBJ=0;
if nargin < 4
    if m<6
        k=c-1;
    else
        k=5;
    end      
end

n=size(X,2);
m0=m;
Success=1;
method=1;
% StartIndZ: before MM update
% EndIndZ: after MM update
if method ==0
    StartIndZ=randsrc(n,1,1:m);
else
    StartIndZ=kmeans(X',m);
end
BiGraph = ones(n,m);
A = meanInd(X, StartIndZ,m,BiGraph);

Ah=[Ah A];
tic
[laKMM, laMM, BiGraph, isCov, obj, ~] = CSBG(X, c, A, k);
laKMMh=[laKMMh laKMM];
ti=toc;
OBJ(1)=obj(end);
% fprintf('time:%d,obj:%d\n',ti,obj)
iter=1;
while(iter<Iter)
    iter = iter +1;
    if isCov
        fprintf('iter:%d\n',iter)
        OBJ(iter)=obj(end);       
%         [Z, ~, ~, id]= ConstructA_NP(X, Anc,k);
%         MidIndZ=id(:,1);
        if (all(StartIndZ==laMM))
        % if OBJ(end)==OBJ(end-1) || (all(StartInd==EndInd) & all(StartIndZ==EndIndZ))
%             Anc = meanInd(X, EndIndZ, c, Z);
            fprintf('all mid=end \n')
            return;
        elseif(length(unique(laMM))~=m)
            fprintf('length(unique(EndIndZ))~=m \n')
            StartIndZ=laMM;
            while(length(unique(StartIndZ))~=m)
                fprintf('len mid ~=m \n')
                A = A(:,unique(StartIndZ));
                m = length(unique(StartIndZ));
                if length(unique(StartIndZ))>c 
                    [BiGraph, ~, ~, id]= ConstructA_NP(X, A,k);
                    StartIndZ=id(:,1);                   
                else % re-ini
                    m=m0;
                    StartIndZ=kmeans(X',m);
                    BiGraph=ones(n,m);
                    A = meanInd(X, StartIndZ, m, BiGraph);
                    Success=0;
                end
            end
            if Success==0
                Ah=[];
            end
            Ah=[Ah A]; Success=1;               
        else
            fprintf('mid ~=end & len min=m \n')
            StartIndZ=laMM;  
            A = meanInd(X, StartIndZ, m, BiGraph);
            Ah=[Ah A];
        end
    else
        fprintf('0~=isCov\n')
        StartIndZ=kmeans(X',m);
        BiGraph=ones(n,m);
        A = meanInd(X, StartIndZ, m, BiGraph);
        Ah=[];
        Ah=[Ah A];
    end
    
    tic
    [laKMM, laMM, BiGraph, isCov, obj, ~] = CSBG(X, c, A, k);
    laKMMh=[laKMMh laKMM];
    ti=toc;
    % fprintf('time:%s,obj:%d\n',ti,obj)
end
fprintf('loop:%d\n',iter)
end

