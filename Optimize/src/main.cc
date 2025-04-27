#include <iostream>
#include <vector>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

#include "CompressedSpMV.hh"

int main()
{
  // 1. Define the dimensions
  Eigen::Index n = 5;
  Eigen::Index m = 3;

  // 2. Create the sparse matrices A (n x n) and B (m x m)
  Eigen::SparseMatrix<double> A(n, n);
  Eigen::SparseMatrix<double> B(m, m);
  Eigen::VectorXd vv;
  vv.segment(0, 1);

  std::vector<Eigen::Triplet<double>> A_triplets;
  std::vector<Eigen::Triplet<double>> B_triplets;
  // Fill with slightly varying diagonal values for a non-trivial example
  for(Eigen::Index i=0; i<n; ++i) A_triplets.push_back(Eigen::Triplet<double>(i, i, 2.0 + i * 0.1));
  for(Eigen::Index i=0; i<m; ++i) B_triplets.push_back(Eigen::Triplet<double>(i, i, 3.0 + i * 0.2));

  A.setFromTriplets(A_triplets.begin(), A_triplets.end());
  B.setFromTriplets(B_triplets.begin(), B_triplets.end());
  A.makeCompressed(); // Good practice for sparse matrix performance
  B.makeCompressed(); // Good practice for sparse matrix performance

  std::cout << "Matrix A (sparse " << A.rows() << "x" << A.cols() << "):\n" << A << std::endl;
  std::cout << "Matrix B (sparse " << B.rows() << "x" << B.cols() << "):\n" << B << std::endl;

  // 3. Create the MyMatrix wrapper object Q
  MyMatrix Q(&A, &B);

  std::cout << "Matrix Q dimensions (matrix-free): " << Q.rows() << "x" << Q.cols() << std::endl;

  Eigen::VectorXd b(n+m), x;
  
  for (Eigen::Index i = 0; i < n+m; ++i) {
    b(i) = 1;
  }
 
  Eigen::VectorXd prod = Q * b;

  std::cout << prod << std::endl;
  
  b.setRandom();
 
  // Solve Ax = b using various iterative solver with matrix-free version:
  {
    Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    cg.compute(Q);
    x = cg.solve(b);
    std::cout << "CG:       #iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;
  }
 
  // {
  //   Eigen::BiCGSTAB<MatrixReplacement, Eigen::IdentityPreconditioner> bicg;
  //   bicg.compute(A);
  //   x = bicg.solve(b);
  //   std::cout << "BiCGSTAB: #iterations: " << bicg.iterations() << ", estimated error: " << bicg.error() << std::endl;
  // }
}