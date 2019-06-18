function [clusternum, label] = struG2la(Z)
    [n,m] = size(Z);
    SS0 = sparse(n+m,n+m); SS0(1:n,n+1:end) = Z; SS0(n+1:end,1:n) = Z';
    [clusternum, label] = graphconncomp(SS0);
    label = label(1:n)';  

end