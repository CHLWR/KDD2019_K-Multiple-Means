
function [laKMM, laMM, BiGraph,isCov, OBJ, alpha, lambda] = CSBG(X, c, A, k, alpha, lambda)
% Input:
%       - X: the data matrix of size nFea x nSmp, where each column is a sample
%               point
%       - c: the number of clusters
%       - A: the matrix of multiple means(MM) of size nFea x nMM
%       - k: the number of neighbor points
% Output:
%       - laKMM: the cluster assignment for each point
%       - laMM: the sub-cluster assignment for each point
%       - BiGraph: the matrix of size nSmp x nMM
% Requre:
% 		ConstructA_NP.m
% 		EProjSimplex_new.m
% 		svd2uv.m
% 		struG2la.m
% Usage:
%       % X: d*n
%       [laKMM, laMM, BiGraph, isCov, obj, ~] = CSBG(X, c, A, k);
% Reference:
%
%	Feiping Nie, Cheng-Long Wang, Xuelong Li, "K-Multiple-Means: A Multiple-Means 
%   Clustering Method with Specified K Clusters," In The 25th ACM SIGKDD Conference
%   on Knowledge Discovery and Data Mining (KDD ¡¯19), August 4¨C8, 2019, Anchorage, AK, USA.
%
%   version 1.0 --May./2019 
%
%   Written by Cheng-Long Wang (ch.l.w.reason AT gmail.com)

NITER = 30;
zr = 10e-5;
if nargin < 4
    k = 5;
end
n = size(X,2);
m = size(A,2);
[Z, Alpha, distX, id] =  ConstructA_NP(X, A,k); % A is sparse
[ZT, AlphaT, distXT, idT] =  ConstructA_NP(A,X,k);

if nargin <5
alpha = 1*mean(Alpha);
alphaT = 1*mean(AlphaT); 
end

 if nargin <6
    lambda = (alpha+alphaT)/2;
 end

Z0 = (Z+ZT')/2;
[BiGraph, U, V, evc, D1, D2] = svd2uv(Z0, c);


if sum(evc(1:c)) > c*(1-zr)
    error('The original graph has more than %d connected component£¬ Please set k larger', c);      
end;

Ater = 0;
dxi = zeros(n,k);
% size(distX),size(id),size(dxi)
for i = 1:n
    dxi(i,:) = distX(i,id(i,:));
end
dxiT = zeros(m,k);
for i = 1:m
    dxiT(i,:) = distXT(i,idT(i,:));
end
OBJ=[];
Ater=0;
for iter = 1:NITER
    U1 = D1*U;
    V1 = D2*V;
    dist = sqdist(U1',V1');  % only local distances need to be computed. speed will be increased using C
    tmp1 = zeros(n,k); 
    for i = 1:n
        dfi = dist(i,id(i,:));
        ad = -(dxi(i,:)+lambda*dfi)/(2*alpha);   
        tmp1(i,:) = EProjSimplex_new(ad);
    end
    Z = sparse(repmat([1:n],1,k),id(:),tmp1(:),n,m);

    tmp2 = zeros(m,k);
    for i = 1:m
        dfiT = dist(idT(i,:),i);
        ad =  (dxiT(i,:)-0.5*lambda*dfiT');        
        tmp2(i,:) = EProjSimplex_new(ad);
    end 
    ZT = sparse(repmat([1:m],1,k),idT(:),tmp2(:),m,n);  

    BiGraph = (Z+ZT')/2;
    U_old = U;
    V_old = V;
    [BiGraph, U, V, evc, D1, D2] = svd2uv(BiGraph, c);
    
%     obj = loss(distX,Z,alpha,lambda,U,V);
%     OBJ=[OBJ obj];
%     fprintf('obj:%6.6f,lambda:%6.6f\n',obj, lambda)

    fn1 = sum(evc(1:c));
    fn2 = sum(evc(1:c+1));

    
    if fn1 < c-zr % the number of block is less than c
        Ater=0;
        lambda = 2*lambda;
    elseif fn2 > c+1-zr % the number of block is more than c
        Ater = 0;
        lambda = lambda/2;   U = U_old; V = V_old;
    else
        Ater=Ater+1;
        if(Ater==2)
            break;
        end
    end
end
fprintf('csbg loop:%d\n',iter)
%% 
laMM=id(:,1);

[clusternum, laKMM] = struG2la(BiGraph);
if clusternum ~=  c
    sprintf('Can not find the correct cluster number: %d', c)
end
isCov=[Ater==2];
obj = loss(distX,BiGraph,alpha,lambda,U,V);
OBJ=[OBJ obj];
end

function obj = loss(distX,Z,alpha,lambda,U,V )
    n = size(Z,1);
    m = size(Z,2);
    a1 = sum(Z,2);
    D1a = spdiags(1./sqrt(a1),0,n,n);  
    a2 = sum(Z,1);
    D2a = spdiags(1./sqrt(a2'),0,m,m);
    st = sum(sum(distX.*Z));
    at = alpha*sum(sum(Z.^2));
    Da = spdiags( [ 1./sqrt(a1) ;1./sqrt(a2')],0,n+m,n+m);
    SS = sparse(n+m,n+m); SS(1:n,n+1:end) = Z; SS(n+1:end,1:n) = Z';
    ft = lambda*trace([U; V]'*(eye(n+m)-Da*SS*Da )*[U; V]);
    obj = st+ at  + ft;

end
