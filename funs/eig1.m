function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, B)
% The optimal solution F to the problem is formed by the c eigenvectors of
% L corresponding to the c smallest eigenvalues
if nargin < 2
    c = size(A,1);
    isMax = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
end;

if nargin < 4
    B = eye(size(A,1));
end;

A = (A+A')/2;   % 这个是啥意思

[v d] = eig(A,B);
d = diag(d);
d = abs(d);
if isMax == 0
    [d1, idx] = sort(d );
else
    [d1, idx] = sort(d,'descend');
end;

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);