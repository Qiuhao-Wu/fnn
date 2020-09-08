function y=ifftshift1(x,dim)
if nargin>1
    if(~isscalar(dim))||floor(dim)~=dim||dim<1
        error(message('MATLAB:ifftshift:DimNotPosInt'))
    end
    m=size(x,dim);
    p=ceil(m/2);
    idx{dim}=[p+1:m 1:p];
else
    numDims=ndims(x);
    idx=cell(1,numDims);
    for k=1:numDims
        m=size(x,k);
        p=ceil(m/2);
        idx{k}=[p+1:m 1:p];
    end
end
y=x(idx{:});