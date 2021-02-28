function B = blkcolfun(A, blksize, f, varargin)
%BLKCOLFUN processes the matrix A by rearranging each m-by-n block of A into
%a column of a temporary matrix, and then applying the function fun to this
%matrix. colfilt zero-pads A, if necessary. Finaly, rearranges the result
%back to get the output.
% 
%   Input:
%   --------
%   - A:        original image, 2D matrix
% 
%   - blksize: size of destination, 2-element int row vector
% 
%   - f:        function handle
% 
%   - varargin: optional inputs, contains the following variables
%               - padding:  whether to pad the boundary blocks or discard 
%                them, logical, true (default) | false
% 
%               - blktype:  block type, str, 'sliding' (default)| 'distinct'
%                           Specified as 'sliding' for sliding neighborhoods
%                           or 'distinct' for distinct blocks.
% 
%   Note:
%   --------
%   For "distinct" mode, the "f" should return a row vector with the number
%   of elements equal to the number of input matrix's columns, or a matrix with the
%   same size as the input matrix. But for "sliding" mode, only the former
%   case is accepted.
% 
%   So, for "distinct" mode, BLKCOLFUN can return a matrix with size equal to
%   the devided blocks' shape or equal to the input matrix's size according to
%   the "f". For "sliding" mode, BLKCOLFUN will return a matrix with size equal to
%   the devided blocks' shape (when "padding" is ture, the size equal to the 
%   input matrix's size, otherwise, it may be smaller as the boundary
%   blocks are discarded.)
% 
%   Output:
%   --------
%   - B:        output matrix
% 
% 
%   Info:
%   --------
%   Created:    Zhihong Zhang <z_zhi_hong@163.com>, 2020-03-29
%   Last Modified:   Zhihong Zhang, 2020-03-29
%
%   Copyright (c) 2020 Zhihong Zhang


%% input setting
padding = 'true';
blktype = 'distinct';

for var_item = varargin
    var_value = var_item{1};
    if islogical(var_value)
        padding = var_value;
    elseif ischar(var_value) || isstring(var_value)
        if isstring(var_value)
            var_value = char(var_value);
        end
        blktype = var_value;
    else
        error("error input")
    end
end


blktype = [lower(blktype) ' ']; % Protect against short string

if isvector(blksize) && numel(blksize)==2
     if ~isrow(blksize)
        blksize = blksize(:)';
     end
elseif isscalar(blksize)
    blksize = [blksize blksize];
else
     error("error input - 'blksize'");
end

%% processing
if blktype(1)=='d'

    % input matrix size
    A_size = size(A);
    
    % Convert neighborhoods of matrix A to columns
    tmp_mat = im2col2(A, blksize, 'distinct');

    % call fun
    tmp_res = f(tmp_mat);

    % rearrange outputs
    if all(size(tmp_res) == size(tmp_mat))
        col_re_size = [blksize(1), blksize(2)];         % col2im's column rearrange's size
        out_re_size = ceil(A_size./blksize).*blksize;   % col2im's output rearrange's size
        out_noborder_sz = A_size;                       % no-border output rearrange's size
        
        B = col2im2(tmp_res, col_re_size, out_re_size, 'distinct');
        if ~padding
            B = B(1:out_noborder_sz(1), 1:out_noborder_sz(2));
        end
    
    elseif isrow(tmp_res)
        
        out_re_size = ceil(A_size./blksize);     % col2im's output rearrange's size
        out_noborder_sz = floor(A_size./blksize); % no-border output rearrange's size
        
%         col_re_size = [1 1];                    % col2im's column rearrange's size
%         B = col2im(tmp_res, col_re_size, out_re_size, 'distinct');
        B = reshape(tmp_res, out_re_size);
        if ~padding
            B = B(1:out_noborder_sz(1), 1:out_noborder_sz(2));
        end        
    end
    
elseif blktype(1)=='s'
    % input matrix size
    A_size = size(A);
    
    % padding
    if padding
        left_up_pos = floor((blksize - 1)/2);
        A_padding = zeros(A_size+blksize - 1);
        A_padding(left_up_pos(1)+1:left_up_pos(1)+A_size(1),left_up_pos(2)+1:...
            left_up_pos(2)+A_size(2)) = A;
        A = A_padding;
        out_re_size = A_size; % col2im's output rearrange's size
    else
        out_re_size = (A_size-blksize)+1; % col2im's output rearrange's size
    end
    
    % Convert neighborhoods of matrix A to columns
    tmp_mat = im2col2(A, blksize, 'sliding');

    % call fun
    tmp_res = f(tmp_mat);

    % rearrange outputs
    if isrow(tmp_res) 
        B = reshape(tmp_res, out_re_size);    
    else
        error("the function should have a row-vector return value");
    end
end
      
end

function b=im2col2(a, block, kind)
%IM2COL2 Rearrange image blocks into columns.
% rewrite from the build-in function IMG2COL 
% 
% See also:
% IMG2COL 

if kind(1)=='d'
        
        nrows = block(1);
        ncols = block(2);
        nElementBlk = nrows*ncols;
    
        mpad = mod(size(a,1),nrows); if mpad>0, mpad = block(1)-mpad; end
        npad = mod(size(a,2),ncols); if npad>0, npad = block(2)-npad; end

        aPad = zeros([size(a,1)+mpad size(a,2)+npad]);
        aPad(1:size(a,1),1:size(a,2)) = a;

        t1 = reshape(aPad,nrows,size(aPad,1)/nrows,[]);
        t2 = reshape(permute(t1,[1 3 2]),size(t1,1)*size(t1,3),[]);
        t3 =  permute(reshape(t2,nElementBlk,size(t2,1)/nElementBlk,[]),[1 3 2]);
        b = reshape(t3,nElementBlk,[]);
    
    
elseif kind(1)=='s'
    [ma,na] = size(a);
    m = block(1); n = block(2);
    
    if any([ma na] < [m n]) % if neighborhood is larger than image
        b = zeros(m*n,0);
        return
    end
    
    % Create Hankel-like indexing sub matrix.
    mc = block(1); nc = ma-m+1; nn = na-n+1;
    cidx = (0:mc-1)'; ridx = 1:nc;
    t = cidx(:,ones(nc,1)) + ridx(ones(mc,1),:);    % Hankel Subscripts
    tt = zeros(mc*n,nc);
    rows = 1:mc;
    for i=0:n-1
        tt(i*mc+rows,:) = t+ma*i;
    end
    ttt = zeros(mc*n,nc*nn);
    cols = 1:nc;
    for j=0:nn-1
        ttt(:,j*nc+cols) = tt+ma*j;
    end
    
    % If a is a row vector, change it to a column vector. This change is
    % necessary when A is a row vector and [M N] = size(A).
    if ismatrix(a) && na > 1 && ma == 1
        a = a(:);
    end
    b = a(ttt);
    
else
    % We should never fall into this section of code.  This problem should
    % have been caught in input parsing.
    error(message('images:im2col2:internalErrorUnknownBlockType', kind));
end
end

function a = col2im2(b,block,mat,kind)
%COL2IM Rearrange matrix columns into blocks.
%   A = COL2IM(B,[M N],[MM NN],'distinct') rearranges each column of B into a
%	rewrite from the build-in function IMG2COL 
% 
%   See also:
%   COL2IMG


if kind(1)=='d' % Distinct
    % Check argument sizes
    [m,n] = size(b);
    if prod(block)~=m, error(message('images:col2im:wrongSize')); end
    
    % Find size of padded A.
    mpad = rem(mat(1),block(1)); if mpad>0, mpad = block(1)-mpad; end
    npad = rem(mat(2),block(2)); if npad>0, npad = block(2)-npad; end
    mpad = mat(1)+mpad; npad = mat(2)+npad;
    if mpad*npad/prod(block)~=n
        error(message('images:col2im:inconsistentSize'));
    end
    
    mblocks = mpad/block(1);
    nblocks = npad/block(2);
    aa = repmat(feval(class(b), 0), [mpad npad]);
    x = repmat(feval(class(b), 0), block);
    rows = 1:block(1); cols = 1:block(2);
    for i=0:mblocks-1
        for j=0:nblocks-1
            x(:) = b(:,i+j*mblocks+1);
            aa(i*block(1)+rows,j*block(2)+cols) = x;
        end
    end
    a = aa(1:mat(1),1:mat(2));
    
elseif kind(1)=='s' % sliding
    a = reshape(b,mat(1)-block(1)+1,mat(2)-block(2)+1);
else
    error(message('images:col2im:unknownBlockType', deblank(kind)))
    
end

end
