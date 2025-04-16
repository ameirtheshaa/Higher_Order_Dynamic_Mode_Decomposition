# Mathematical Formulation of Tensor DMD using HOSVD

This document provides the formal mathematical foundation for the Tensor-based Dynamic Mode Decomposition (DMD) method implemented in this library.

## 1. Standard DMD (Matrix Version)

In standard DMD, we consider a sequence of data vectors $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_m \in \mathbb{R}^n$, where each vector represents the state of a dynamical system at a specific time. We assume that these states evolve according to a linear operator $\mathbf{A}$:

$$\mathbf{x}_{k+1} = \mathbf{A} \mathbf{x}_k$$

We arrange the data into two matrices, offset by one time step:

$$\mathbf{X} = 
\begin{bmatrix} 
\mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_{m-1}
\end{bmatrix}$$

$$\mathbf{Y} = 
\begin{bmatrix} 
\mathbf{x}_2 & \mathbf{x}_3 & \cdots & \mathbf{x}_{m}
\end{bmatrix}$$

These matrices satisfy the relation $\mathbf{Y} \approx \mathbf{A} \mathbf{X}$. The DMD operator $\mathbf{A}$ can be approximated as:

$$\mathbf{A} \approx \mathbf{Y} \mathbf{X}^{\dagger}$$

where $\mathbf{X}^{\dagger}$ is the Moore-Penrose pseudoinverse of $\mathbf{X}$.

To compute the DMD, we first perform the SVD of $\mathbf{X}$:

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^*$$

Then, we compute a reduced representation of $\mathbf{A}$:

$$\tilde{\mathbf{A}} = \mathbf{U}^* \mathbf{A} \mathbf{U} = \mathbf{U}^* \mathbf{Y} \mathbf{V} \mathbf{\Sigma}^{-1}$$

We then compute the eigendecomposition of $\tilde{\mathbf{A}}$:

$$\tilde{\mathbf{A}} \mathbf{W} = \mathbf{W} \mathbf{\Lambda}$$

where $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_r)$ contains the eigenvalues, and $\mathbf{W}$ contains the eigenvectors.

The DMD modes $\mathbf{\Phi}$ are given by:

$$\mathbf{\Phi} = \mathbf{Y} \mathbf{V} \mathbf{\Sigma}^{-1} \mathbf{W}$$

## 2. Tensor Notation and Operations

Let $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ be an $N$-way tensor. We denote the elements of $\mathcal{X}$ as $x_{i_1, i_2, \ldots, i_N}$.

The mode-$n$ unfolding (matricization) of a tensor $\mathcal{X}$, denoted by $\mathbf{X}_{(n)}$, rearranges the elements of $\mathcal{X}$ into a matrix by keeping the $n$-th mode as rows and flattening all other modes into columns.

The mode-$n$ product of a tensor $\mathcal{X}$ with a matrix $\mathbf{U} \in \mathbb{R}^{J \times I_n}$, denoted by $\mathcal{X} \times_n \mathbf{U}$, is defined elementwise as:

$$(\mathcal{X} \times_n \mathbf{U})_{i_1, \ldots, i_{n-1}, j, i_{n+1}, \ldots, i_N} = \sum_{i_n=1}^{I_n} x_{i_1, \ldots, i_N} \cdot u_{j, i_n}$$

## 3. Higher-Order SVD (Tucker Decomposition)

The Tucker decomposition of an $N$-way tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is given by:

$$\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \cdots \times_N \mathbf{U}^{(N)}$$

where:
- $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times \cdots \times R_N}$ is the core tensor
- $\mathbf{U}^{(n)} \in \mathbb{R}^{I_n \times R_n}$ is the factor matrix for the $n$-th mode
- $R_n \leq I_n$ is the rank of the $n$-th mode
- $\times_n$ denotes the mode-$n$ product

The factor matrices $\mathbf{U}^{(n)}$ are usually orthogonal and can be computed by finding the left singular vectors of the mode-$n$ unfolding of $\mathcal{X}$:

$$\mathbf{X}_{(n)} = \mathbf{U}^{(n)} \mathbf{\Sigma}^{(n)} {\mathbf{V}^{(n)}}^T$$

The core tensor $\mathcal{G}$ is then computed as:

$$\mathcal{G} = \mathcal{X} \times_1 {\mathbf{U}^{(1)}}^T \times_2 {\mathbf{U}^{(2)}}^T \times_3 \cdots \times_N {\mathbf{U}^{(N)}}^T$$

## 4. Tensor-based DMD using HOSVD

Let $\mathcal{X} \in \mathbb{R}^{M \times I_1 \times I_2 \times \cdots \times I_N}$ be a tensor where the first dimension represents time (with $M$ snapshots), and the remaining dimensions represent spatial coordinates.

### 4.1 Data Splitting

We split the tensor into two parts, offset by one time step:

$$\mathcal{X}_1 = \mathcal{X}_{1:M-1, :, :, \ldots, :} \in \mathbb{R}^{(M-1) \times I_1 \times I_2 \times \cdots \times I_N}$$

$$\mathcal{X}_2 = \mathcal{X}_{2:M, :, :, \ldots, :} \in \mathbb{R}^{(M-1) \times I_1 \times I_2 \times \cdots \times I_N}$$

### 4.2 HOSVD Dimensionality Reduction

We apply HOSVD to $\mathcal{X}_1$:

$$\mathcal{X}_1 \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \cdots \times_{N+1} \mathbf{U}^{(N+1)}$$

where:
- $\mathbf{U}^{(1)} \in \mathbb{R}^{(M-1) \times R_1}$ corresponds to the temporal mode
- $\mathbf{U}^{(n+1)} \in \mathbb{R}^{I_n \times R_{n+1}}$ corresponds to the $n$-th spatial mode
- $R_n$ are the chosen ranks for each mode

### 4.3 Projection onto Reduced Space

We project both $\mathcal{X}_1$ and $\mathcal{X}_2$ onto the reduced space:

$$\tilde{\mathcal{X}}_1 = \mathcal{X}_1 \times_2 {\mathbf{U}^{(2)}}^T \times_3 {\mathbf{U}^{(3)}}^T \times_4 \cdots \times_{N+1} {\mathbf{U}^{(N+1)}}^T$$

$$\tilde{\mathcal{X}}_2 = \mathcal{X}_2 \times_2 {\mathbf{U}^{(2)}}^T \times_3 {\mathbf{U}^{(3)}}^T \times_4 \cdots \times_{N+1} {\mathbf{U}^{(N+1)}}^T$$

### 4.4 Matricization for DMD

We unfold $\tilde{\mathcal{X}}_1$ and $\tilde{\mathcal{X}}_2$ along the first mode (time) to obtain matrices:

$$\tilde{\mathbf{X}}_1 = [\tilde{\mathcal{X}}_1]_{(1)} \in \mathbb{R}^{(M-1) \times (R_2 \times R_3 \times \cdots \times R_{N+1})}$$

$$\tilde{\mathbf{X}}_2 = [\tilde{\mathcal{X}}_2]_{(1)} \in \mathbb{R}^{(M-1) \times (R_2 \times R_3 \times \cdots \times R_{N+1})}$$

### 4.5 Compute DMD Operator

The reduced DMD operator $\tilde{\mathbf{A}}$ is computed as:

$$\tilde{\mathbf{A}} = \tilde{\mathbf{X}}_2 \tilde{\mathbf{X}}_1^{\dagger}$$

### 4.6 Eigendecomposition

We compute the eigendecomposition of $\tilde{\mathbf{A}}$:

$$\tilde{\mathbf{A}} \mathbf{W} = \mathbf{W} \mathbf{\Lambda}$$

where $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_r)$ contains the eigenvalues, and $\mathbf{W}$ contains the eigenvectors.

### 4.7 DMD Modes in Reduced Space

The DMD modes in the reduced space are given directly by the eigenvectors:

$$\tilde{\mathbf{\Phi}} = \mathbf{W}$$

### 4.8 DMD Modes in Original Space

To transform the DMD modes back to the original space, we first reshape each column of $\tilde{\mathbf{\Phi}}$ into a tensor of shape $(R_2 \times R_3 \times \cdots \times R_{N+1})$:

$$\tilde{\mathcal{\Phi}}_j = \text{reshape}(\tilde{\mathbf{\Phi}}_{:,j}, [R_2, R_3, \ldots, R_{N+1}])$$

Then, we transform each mode tensor back to the original space:

$$\mathcal{\Phi}_j = \tilde{\mathcal{\Phi}}_j \times_1 \mathbf{U}^{(2)} \times_2 \mathbf{U}^{(3)} \times_3 \cdots \times_N \mathbf{U}^{(N+1)}$$

### 4.9 DMD Frequencies

The continuous-time DMD frequencies are computed as:

$$\omega_j = \frac{\ln(\lambda_j)}{\Delta t}$$

where $\Delta t$ is the time step between consecutive snapshots.

## 5. Reconstruction and Prediction

To reconstruct or predict the tensor's evolution, we need to compute the mode amplitudes by projecting the initial condition onto the DMD modes.

### 5.1 Initial Condition in Reduced Space

Let $\mathbf{x}_0$ be the initial condition (first time slice of $\mathcal{X}$):

$$\mathbf{x}_0 = \mathcal{X}_{1,:,:,\ldots,:}$$

We project it onto the reduced space:

$$\tilde{\mathbf{x}}_0 = \mathbf{x}_0 \times_1 {\mathbf{U}^{(2)}}^T \times_2 {\mathbf{U}^{(3)}}^T \times_3 \cdots \times_N {\mathbf{U}^{(N+1)}}^T$$

### 5.2 Mode Amplitudes

We flatten $\tilde{\mathbf{x}}_0$ into a vector and compute the mode amplitudes by solving:

$$\tilde{\mathbf{\Phi}} \mathbf{b} = \text{vec}(\tilde{\mathbf{x}}_0)$$

where $\mathbf{b}$ contains the mode amplitudes.

### 5.3 Reconstruction and Prediction

For any time $t$, the state can be reconstructed as:

$$\tilde{\mathbf{x}}(t) = \sum_{j=1}^{r} b_j e^{\omega_j t} \tilde{\mathbf{\Phi}}_{:,j}$$

Transforming back to the original space:

$$\mathbf{x}(t) = \text{reshape}(\tilde{\mathbf{x}}(t), [R_2, R_3, \ldots, R_{N+1}]) \times_1 \mathbf{U}^{(2)} \times_2 \mathbf{U}^{(3)} \times_3 \cdots \times_N \mathbf{U}^{(N+1)}$$

## 6. Implementation Notes

In practical implementations, numerical stability can be improved by:

1. Using truncated SVD for computing the factor matrices
2. Choosing appropriate ranks $R_n$ for each mode
3. Regularizing the DMD operator computation
4. Using randomized algorithms for large-scale problems

## References

1. Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM.
2. Klus, S., Gelß, P., Peitz, S., & Schütte, C. (2018). *Tensor-based dynamic mode decomposition*. Nonlinearity, 31(7), 3359.
3. De Lathauwer, L., De Moor, B., & Vandewalle, J. (2000). *A multilinear singular value decomposition*. SIAM Journal on Matrix Analysis and Applications, 21(4), 1253-1278.