function [Z, Alpha, Dis, id, tmp]= ConstructA_NP(A, B, k, isSparse)
% d*n
% Z is sparse
% writed by Cheng-Long Wang ,mail:ch.l.w.reason@gmail.com

if nargin<4
    isSparse = 1;
end
if nargin<3
    k = 5;
end
n = size(A,2);
if isempty(B) || n==size(B,2)
    B = A;
    m=n;
    if n>10000
        block_size = 10;
        save_type = 3;
        Dis = gen_nn_distanceA(A', k+1, block_size, save_type);
        distXt = Dis;
        di = zeros(n,k+1);
        id = di;  
        for i = 1:k+1
            [di(:,i),id(:,i)] = max(distXt, [], 2);
            temp = (id(:,i)-1)*n+[1:n]';
            distXt(temp) = 0;
        end
        id=fliplr(id);
        di=fliplr(di);
    else
        Dis = sqdist(A,B); % O(ndm)
        distXt = Dis;
        di = (zeros(n,k+2));
        id = di;  
        for i = 1:k+2
            [di(:,i),id(:,i)] = min(distXt, [], 2);
            temp = (id(:,i)-1)*n+[1:n]';
            distXt(temp) = 1e100;
        end
        di(:,1) = [];
        id(:,1) = [];
    end
else
    Dis = sqdist(A,B);
    distXt = Dis;
    di = zeros(n,k+1);
    id = di;  
    for i = 1:k+1
        [di(:,i),id(:,i)] = min(distXt, [], 2);
        temp = (id(:,i)-1)*n+[1:n]';
        distXt(temp) = 1e100;
    end
end

m = size(B,2);
clear distXt temp
id(:,end) = [];

Alpha = 0.5*(k*di(:,k+1)-sum(di(:,1:k),2)); 
ver=version;
if(str2double(ver(1:3))>=9.1)
    tmp = (di(:,k+1)-di(:,1:k))./(2*Alpha+eps); % for the newest version(>=9.1) of MATLAB
else
    tmp =  bsxfun(@rdivide,bsxfun(@minus,di(:,k+1),di(:,1:k)),2*Alpha+eps); % for old version(<9.1) of MATLAB
end
Z = sparse(repmat([1:n],1,k),id(:),tmp(:),n,m);
if ~isSparse
    Z=full(Z);
end
return
