
function [Z, U, V, evc, D1z, D2z] = svd2uv(Z, c)
    [n,m] = size(Z);
    ver=version;
    if(str2double(ver(1:3))>=9.1)
        Z = Z./sum(Z,2);% for the newest version(>=9.1) of MATLAB
    else
        Z = bsxfun(@rdivide, Z, sum(Z,2));% for old version(<9.1) of MATLAB
    end
    z1 = sum(Z,2);
    D1z = spdiags(1./sqrt(z1),0,n,n);  
    z2 = sum(Z,1);
    D2z = spdiags(1./sqrt(z2'),0,m,m);
    Z1 = D1z*Z*D2z;
    
    LZ = full(Z1'*Z1);

    [V, evc, ~]=eig1(LZ,c+1);
    V = V(:,1:c);
    U = (Z1*V)./(ones(n,1)*sqrt(evc(1:c))');

%     [U, evc, V] = mySVD(Z1,c+1); 
%     V = V(:,1:c);
%     U = U(:,1:c);

    % opts.tol = 1e-8;
    % opts.maxit = 150;
    % [U, evc, V]  = svds(Z1,c+1,'L',opts);evc=spdiags(evc);evc=full(evc);  
    % V = V(:,1:c);
    % U = U(:,1:c);

    U = sqrt(2)/2*U; V = sqrt(2)/2*V;

end
