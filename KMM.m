function [laKMM, Z,Anchors,Sinit, obj] = KMM(X, c, m, k)
% [laKMM, BiGraph, Anchors, ~, ~]= KMM(X', c, m,k) : K-Multiple-Means
% Input:
%       - X: the data matrix of size nFea x nSmp, where each column is a sample
%               point
%       - c: the number of clusters
%       - m: the number of multiple means(MM)
%       - k: the number of neighbor points
% Output:
%       - laKMM: the cluster assignment for each point
%       - Z(BiGraph): the matrix of size nSmp x nMM
%       - Anchors: the multiple means matrix of size nFea x nMM
%       - Sinit: initial BiGraph
% Requre:
%   L2_distance_1.m
%   ConstructA_NP.m
%   EProjSimplex_new.m
%   eig1.m
% Usage:
%       % X: d*n
%       [laKMM,BiGraph,Anchors]= KMM(X, c, m,k) ;
% Reference:
%
%	Feiping Nie, Cheng-Long Wang, Xuelong Li, "K-Multiple-Means: A Multiple-Means 
%   Clustering Method with Specified K Clusters," In The 25th ACM SIGKDD Conference
%   on Knowledge Discovery and Data Mining (KDD ’19), August 4–8, 2019, Anchorage, AK, USA.
%
%   version 1.0 --Jan./2019 
%
%   Written by Cheng-Long Wang (ch.l.w.reason AT gmail.com)

NITER = 31;
Lose=0;
RITER = 31;
zr = 10e-7;
flag=1;
lambda_ = 10e-2;
step=2;
Solver = 1;
if nargin < 4
    k = 5;
end
n=size(X,2);
m0=m;
idm=randperm(n,m);
Anchors = X(:,idm); 

[A0, Gamma, distX, id]= ConstructA_NP(X, Anchors,k);
[AT0, GammaT, distXT, idT]= ConstructA_NP(Anchors,X,k);
label=id(:,1); % nearest anchor index
while( length(unique(label))~=m ) % if unique(nearest anchor index) !=m,update m
    fprintf('length(unique(label))~=m, anchor update\n')
    anchors_ = Anchors(:,unique(label));
    m=size(anchors_,2);
    if m > c % if after update m, m>c
        [A0, Gamma, distX, id]= ConstructA_NP(X, anchors_,k);
        [AT0, GammaT, distXT, idT]= ConstructA_NP(anchors_,X,k);
        label=id(:,1);
        Anchors = anchors_;
    else % if after update m, m<=c, re-select anchorss
        m=m0;
        randId = randperm(n,m);
        anchors_new = X(:,randId);
        [A0, Gamma, distX, id]= ConstructA_NP(X, anchors_new,k);
        [AT0, GammaT, distXT, idT]= ConstructA_NP(anchors_new,X,k);
        label=id(:,1);
        Anchors = anchors_new;
    end
end
 
gamma = 1*mean(Gamma);
gammaT = 1*mean(GammaT);

if flag==1
    lambda=mean(gamma+gammaT);
else
    lambda=lambda_;
end
A=(A0+AT0')/2; % k-nearest neighbor graph, A0:k-nearest-anchor for data, AT0:k-nearest-data_points for anchor
%% 
n = size(A,1);
m = size(A,2);
A = sparse(A);
A = A./sum(A,2);
Sinit = A;
a1 = sum(A,2);
D1a = spdiags(1./sqrt(a1),0,n,n);  
a2 = sum(A,1);
D2a = spdiags(1./sqrt(a2'),0,m,m);
Sinit1 = D1a*A*D2a;

SS2 = Sinit1'*Sinit1;
SS2 = full(SS2); 
[V,ev0,ev]=eig1(SS2,m); % O(m*m*k)

V = V(:,1:c); % find first c maximum eigenvector & eigenvalue;
U=(Sinit1*V)./(ones(n,1)*sqrt(ev0(1:c))');
U = sqrt(2)/2*U; V = sqrt(2)/2*V;

% if sum(ev(1:c+1)) > (c+1)*(1-zr)
%     error('The original graph has more than %d connected component', c);
% end;

if (sum(ev(1:c)) > c*(1-zr)) && (sum(ev(1:c+1)) < (c)*(1+zr)) % clustered into exactly c clusters fortunately after initialization
    Z=A;
    SS0=sparse(n+m,n+m); SS0(1:n,n+1:end)=Z; SS0(n+1:end,1:n)=Z';
    [~, y1]=graphconncomp(SS0);
    laKMM=y1(1:n);
    laKMM = laKMM(:);
    fprintf('After partition update, Convergence\n')
    return;
end  
old_RankConstr = 0;
%% 
D1 = 1; D2 = 1;
for Rter=1:RITER
    for iter = 1:NITER
        if iter==NITER
            RankConstr=0;
            fprintf('NITER,RankConstr=0\n') % after n iterations, not satisfy the rank constraint
            break;
        end
        %% Update S
        U1 = D1*U;
        V1 = D2*V;
        dist = L2_distance_1(U1',V1');  % actually only local distances need to be computed. speed will be increased using C
%% Solver: A simplex ...   
        if Solver==1
            Z=zeros(n,m);   
            for i=1:n
                dfi = dist(i,id(i,:));
                dxi = distX(i,id(i,:));
                ad = -(dxi+lambda*dfi)/(2*gamma);
                tm=sum(sum(isnan((ad))));
                if tm~=0
                    fprintf('************************%0.5g,%0.5g,%0.5g,%0.5g,%0.5g\n',tm,lambda,gamma,gammaT,length(unique(id)))
                end            
                Z(i,id(i,:)) = EProjSimplex_new(ad);
                tm=sum(sum(isnan((Z(i,id(i,:))))));
                if tm~=0
                    fprintf('*******************%0.5g\n',tm)
                end
            end
            ZT = zeros(m,n);
            for i=1:m
                idxT = idT(i,1:k);
                dxiT = distXT(i,idxT);
                dfiT = dist(idxT,i);
                %ZT:  learn a structural graph from initial input, rather
                %than learn a nearest-neighbor graph for anchor which may
                %destory the structure of graph Z, ref:Feiping Nie 2017NIPS
                %co-clustering
                ad = (dxiT-0.5*lambda*dfiT');  
                ZT(i,idxT) = EProjSimplex_new(ad);
            end   
        end
        %% Update F   
        Z=sparse(Z);
        ZT=sparse(ZT);
        A=(Z+ZT')/2;
        A = A./sum(A,2);
        d1 = sum(A,2);
        D1 = spdiags(1./sqrt(d1),0,n,n);  
        d2 = sum(A,1);
%         m = size(A,2);
        D2 = spdiags(1./sqrt(d2'),0,m,m);
        SS1 = D1*A*D2;
        SS2 = SS1'*SS1; 
        SS2 = full(SS2); 
    %     [V, ev0, ev]=eig1(SS2,c);
        [V,ev0,ev]=eig1(SS2,m); % 
        V = V(:,1:c);
        U=(SS1*V)./(ones(n,1)*sqrt(ev0(1:c))');
        U = sqrt(2)/2*U; V = sqrt(2)/2*V;

        U_o = U;
        V_o = V;
        fn1 = sum(ev(1:c));
        fn2 = sum(ev(1:c+1));
        if fn1<c-zr % clusters are less than c
             Cov=0; % not convergence
            lambda = step*lambda;
        elseif fn2 > c+1-zr  % clusters are larger than c
             Cov=0;
%             fprintf('lambda/2')
            lambda = lambda/(step*0.75);   U = U_o; V = V_o;
        else  
            RankConstr=1; % satisfy rank constraint
            old_RankConstr=1;
            A_old = A;
            distX_old = distX;
            U_old = U;
            V_old = V;
            Anchors_old = Anchors;
            gamma_old = gamma;
            lambda_old = lambda;
            break;
        end
%         fprintf('loop iter %d\n',iter);
    end
    % NITER end, check RankConstr
    if (RankConstr==1)
%         F=[U; V]; SS0=sparse(n+m,n+m);SS0 = [zeros(n),A;(A'),zeros(m)];
%         DD0=sparse(diag(1./sqrt(sparse(sum(SS0,2))))); 
% 
%         st(Rter) = full(sum(sum(distX.*full(A))));
%         at(Rter) =full( sum(sum(gamma*full(A).^2)));
%         tmp1=DD0*SS0*DD0;
%         I=sparse(eye(n+m));
%         tmp=(I-tmp1);
%         ft(Rter) = trace(F'*tmp*F);
%         ft2(Rter) = full(lambda*ft(Rter));
%         obj(Rter) = st(Rter)+ at(Rter)  + lambda*ft(Rter);
%         fprintf('%0.5g,%0.5g,%0.5g,%0.5g,%0.5g\n', st(Rter), at(Rter), ft(Rter), ft2(Rter), obj(Rter) );
        for i=1:m
            sub_idx=find(label==i);
            if length(sub_idx)==1
                Anchors(:,i)=X(:,sub_idx);
            elseif sum(A(sub_idx,i))==0
                Anchors(:,i)=X(:,sub_idx)*ones(length(sub_idx),1)/length(sub_idx);
            else
                Anchors(:,i)=X(:,sub_idx)*A(sub_idx,i)/sum(A(sub_idx,i));
            end  
        end

        [Aup, Gamma, distX, id]= ConstructA_NP(X, Anchors,k);
        [ATup, GammaT, distXT, idT]= ConstructA_NP(Anchors,X,k);
        label_new=id(:,1);
        if ( all(label==label_new)) % partition convergence
            fprintf('partition Convergence\n')
            break;   
        elseif  Rter<RITER % partition not convergencce
            while( length(unique(label_new))~=m ) % check for length(unique())
                fprintf('length(unique(label_new))~=m, anchor_new update\n')
                anchors_new = Anchors(:,unique(label_new));
                m=size(anchors_new,2);
                if m > c
                    [Aup, Gamma, distX, id]= ConstructA_NP(X, anchors_new,k);
                    [ATup, GammaT, distXT, idT]= ConstructA_NP(anchors_new,X,k);
                    label_new=id(:,1);
                    Anchors = anchors_new;
                else
                    m=m0;
                    randId = randperm(n,m);
                    anchors_new = X(:,randId);
                    [Aup, Gamma, distX, id]= ConstructA_NP(X, anchors_new,k);
                    [ATup, GammaT, distXT, idT]= ConstructA_NP(anchors_new,X,k);
                    label_new=id(:,1);
                    Anchors = anchors_new;
                end
            end
            label=label_new;
            fprintf('partition update\n');
            gamma = 1*mean(Gamma);
            gammaT = 1*mean(GammaT);
            if flag==1
                lambda=mean(gamma+gammaT);
            else
                lambda=lambda_;
            end
            Sup=(Aup+ATup')/2;
            %% 
            n = size(Sup,1);
            m = size(Sup,2);
            Sup = sparse(Sup);
            A = Sup./sum(Sup,2);
            a1 = sum(A,2);
            D1a = spdiags(1./sqrt(a1),0,n,n);  
            a2 = sum(A,1);
            D2a = spdiags(1./sqrt(a2'),0,m,m);
            Sup1 = D1a*A*D2a;
            SS2 = Sup1'*Sup1;
            SS2 = full(SS2); 
%             fprintf('m:%0.5g,c:%0.5g\n',m,c)
            [V,ev0,ev]=eig1(SS2,m); % find first c maximum eigenvector & eigenvalue; O(m*m*k)
            V=V(:,1:c);
            U=(Sup1*V)./(ones(n,1)*sqrt(ev0(1:c))');
            U = sqrt(2)/2*U; V = sqrt(2)/2*V;
            D1 = 1; D2 = 1;
        %     if sum(ev(1:c+1)) > (c+1)*(1-zr)
        %     error('The original graph has more than %d connected component', c);
        %     end
            if (sum(ev(1:c)) > c*(1-zr)) && (sum(ev(1:c+1)) < (c)*(1+zr))
                Z=A;
                SS0=sparse(n+m,n+m); SS0(1:n,n+1:end)=Z; SS0(n+1:end,1:n)=Z';
                [~, y1]=graphconncomp(SS0);
                laKMM=y1(1:n);
                laKMM = laKMM(:);
                fprintf('After partition update, Convergence\n')
                return;
            end
        end     
    elseif old_RankConstr==1  % A->RankConstr ~= 1 && A_old->old_RankConstr==1, need to go back
        A=A_old;    % when Rter>1, A_old->RankConstr == 1
        Anchors = Anchors_old;
        distX = distX_old;
        U = U_old;
        V = V_old;
        gamma = gamma_old;
        lambda = lambda_old;
        fprintf('A->RankConstr ~=1 & A_old->RankConstr == 1, A:back to A_old\n')
        break;
    elseif Rter<RITER   % A_0->RankConstr ~= 1 & A_old->RankConstr ~=1
        fprintf('after NITER, A_0->RankConstr ~= 1, re-initialize Anchor \n')
        randId = randperm(n,m);
        Anchors = X(:,randId);
        [Aup, Gamma, distX, id]= ConstructA_NP(X, Anchors,k);
        [ATup, GammaT, distXT, idT]= ConstructA_NP(Anchors,X,k);
        gamma = 1*mean(Gamma);
        gammaT = 1*mean(GammaT);
        if flag==1
            lambda=mean(gamma+gammaT);
        else
            lambda=lambda_;
        end
        label=id(:,1);
        while( length(unique(label))~=m)
%             fprintf('length(unique(label))~=m, anchor update\n')
            anchors_ = Anchors(:,unique(label));
            m=size(anchors_,2);
            if m > c
                [Aup, Gamma, distX, id]= ConstructA_NP(X, anchors_,k);
                [ATup, GammaT, distXT, idT]= ConstructA_NP(anchors_,X,k);
                label=id(:,1);
                Anchors = anchors_;
            else
                m=m0;
                randId = randperm(n,m);
                anchors_new = X(:,randId);
                [Aup, Gamma, distX, id]= ConstructA_NP(X, anchors_new,k);
                [ATup, GammaT, distXT, idT]= ConstructA_NP(anchors_new,X,k);
                label=id(:,1);
                Anchors = anchors_new;
            end           
        end       
%         fprintf('partition update\n');
        Sup=(Aup+ATup')/2;
        %% 
        n = size(Sup,1);
        m = size(Sup,2);
        A = sparse(Sup);        
        A = A./sum(A,2);
        a1 = sum(A,2);
        D1a = spdiags(1./sqrt(a1),0,n,n);  
        a2 = sum(A,1);
        D2a = spdiags(1./sqrt(a2'),0,m,m);
        Sup1 = D1a*A*D2a;
        
        SS2 = Sup1'*Sup1;
        SS2 = full(SS2); 
        [V,ev0,ev]=eig1(SS2,m); % find first c maximum eigenvector & eigenvalue; O(m*m*k)
        V=V(:,1:c);
        U=(Sup1*V)./(ones(n,1)*sqrt(ev0(1:c))');
        U = sqrt(2)/2*U; V = sqrt(2)/2*V;
        D1 = 1; D2 = 1;
    %     if sum(ev(1:c+1)) > (c+1)*(1-zr)
    %     error('The original graph has more than %d connected component', c);
    %     end
        if (sum(ev(1:c)) > c*(1-zr)) && (sum(ev(1:c+1)) < (c)*(1+zr))
            Z=A;
            SS0=sparse(n+m,n+m); SS0(1:n,n+1:end)=Z; SS0(n+1:end,1:n)=Z';
            [~, y1]=graphconncomp(SS0);
            laKMM=y1(1:n);
            laKMM = laKMM(:);
            fprintf('After partition update, Convergence\n')
            return;
        end            
    end
%% Update Anchors
end
if Rter==RITER
    fprintf('RITER, NO Convergence \n')
end 
%% 
Z=A;
m=size(Z,2);
SS0=sparse(n+m,n+m); SS0(1:n,n+1:end)=Z; SS0(n+1:end,1:n)=Z';
[~, y1]=graphconncomp(SS0);
laKMM=y1(1:n);
laKMM = laKMM(:);
% size(distX)
% size(A)
obj = objection(distX,A,gamma,lambda,U,V);

end
function obj = objection(distX,A,gamma,lambda,U,V )
    n=size(A,1);
    m=size(A,2);
    a1 = sum(A,2);
%     D1a = spdiags(1./sqrt(a1),0,n,n);  
    a2 = sum(A,1);
%     D2a = spdiags(1./sqrt(a2'),0,m,m);
    st = sum(sum(distX.*A));
    at = sum(sum(gamma*A.^2));
%     ft = 2*lambda*trace(U' * D1a * A *D2a *V);
        Da = spdiags( [ 1./sqrt(a1) ;1./sqrt(a2')],0,n+m,n+m);
    SS = sparse(n+m,n+m); SS(1:n,n+1:end) = A; SS(n+1:end,1:n) = A';
%     ft = lambda*(trace([U; V]'*eye(n+m)*[U; V])-2*trace(U' * D1a * Z *D2a *V));
    ft = lambda*trace([U; V]'*(eye(n+m)-Da*SS*Da )*[U; V]);
    
%     ft2 = lambda*ft(iter);
    obj = st+ at  + ft;

end


