function means = meanInd(X, label, c, Z)
% X:d*n, Z:n*c
n=size(X,2);
if length(unique(label))~=c
	label = kmeans(X',c);
	Z = ones(n,c);
% 	fprintf('n ')
end
for i=1:c
    sub_idx=find(label==i);
    means(:,i)=X(:,sub_idx)*Z(sub_idx,i)/sum(Z(sub_idx,i));
end

end