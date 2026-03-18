
# General set-up

Assume tensors have layouts
- I[G,N,C,Hi,Wi]
- W[G,K,C,Y,X]
- O[G,N,K,Ho,Wo]

for a 2D forward convolution problem defined by

- N = input batch size
- G = number of conv groups
- C = number of input filters per group
- K = number of output filters per group
- Hi = input height (corresponds y-direction)
- Wi = input width (corresponds to x-direction)
- Y = filter height
- X = filter width
- Ho = output height (corresponds to y-direction)
- Wo = output width (corresponds to x-direction)
- Sy = stride in y-direction 
- Sx = stride in x-direction
- Dy = dilation in y-direction
- Dx = dilation in x-direction  
- LPy = left padding in y-direction 
- Lpx = left padding in x-direction 
- RPy = right padding in y-direction 
- RPx = right padding in x-direction

The Im2Win transformation maps the multi-dimensional problem problem into a GEMM problem of size

- Mgemm = K
- Ngemm = N x Ho x Wo
- Kgemm = C x Y x X

## Im2Win transformation

The Im2Win transformation maps the original tensor I to im2win tensor I' via relation

I'(i, r, m, k * Y + u) = I(i, r, m + u, n + v),

where 

m = 0, ..., Ho - 1
n = 0, ..., Wo - 1
u = 0, ..., Y - 1
v = 0, ..., X - 1
i = 0, ..., N - 1
r = 0, ..., C - 1
k = 0, ..., Wi - 1

this gives mapping I' = im2win(I, W)

## Creating im2win tensor explicitly

Assume we have the initial input tensor I in the channels-first layout I[N,C,Hi,Wi].
We can create the im2win tensor explicitly by allocating a new tensor I' of size (H, C, Ho,Wi x Y) and running 
the 5-fold loop to fill the values

for i=0 to Ni−1 in parallel do 
 for r=0 to Ci−1 do 
  for m=0 to Ho−1 do 
   for k=0 to Wi−1 do 
    for u=0 to Hf−1 do 
      I'[i][r][m][k∗Hf+u]= I[i][r][m∗s+u][k]

Once we constructed I', we can write the convolution in terms of the index calculations described in the next section.

## Explicit formulas for implicit im2win

Let's skip the convolution group dimension G as it corresponds to batched GEMM batch index.
Let (m,n,k) be the indices corresponding to GEMM MNK dimensions of size (Mgemm, Ngemm, Kgemm).

We can write the tensor indices as functions of (m,n,k) as follows

- O(o_n, o_c, o_h, o_w)
- W(f_k, f_c, f_h, f_w)
- I'(i_n, i_c, i_h, i_w)

where the convolution tensor indices are expressed by (m,n,k) via equations

- o_n = n / (Ho x Wo)
- o_c = m
- o_h = [n % (Ho x Wo)] / Wo
- o_w = [n % (Ho x Wo)] % Wo

- f_k = m
- f_c = k / (Y x X)
- k_res = k % (Y x X)
- f_h = k_res / X
- f_w = k_res % X

- i_n = n / (Ho x Wo)
- i_c = k / (Y x X)
- i_h = o_h x s_h + f_h x d_h - p_h
- i_w = o_w x s_w + f_w x d_w - p_w

where (s_h, s_w) is the stride, (d_h, d_w) is dilation, and (p_h, p_w) is the padding.

## Tiled im2win 

We can re-use the CK Tile implicit GEMM pipelines to implement direct convolution

O(o_n, o_c, o_h, o_w) = sum_k I'(i_n, i_c, i_h, i_w) * W(f_n, f_c, f_h, f_w)

Hence, we replace the im2col transformation of the implicit GEMM by the transformation defined in the section above.
This can be implemented as a tensor descriptor.

Here k runs from zero to Kgemm = C * Y * X.
