#pragma once 

#include <iostream>
#include <vector>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

struct PrimalMatrix {
  using Index = Eigen::Index;
  const Eigen::SparseMatrix<double>* pmat_M;
  const Eigen::VectorXd* pvec_C;  // col vec
  const Eigen::VectorXd* pvec_VQ; // col vec
  const Index size_Q;
  const Index size_M;
  const Index size_N;

  PrimalMatrix(Index size_Q, Index size_N, Index size_M, 
               const Eigen::SparseMatrix<double>* mat_M_ptr,
               const Eigen::VectorXd* vec_C_ptr,
               const Eigen::VectorXd* vec_VQ_ptr)
              : size_Q{ size_Q }, size_M{ size_M }, size_N{ size_N },
                pmat_M{ mat_M_ptr }, 
                pvec_C{ vec_C_ptr }, 
                pvec_VQ{ vec_VQ_ptr }
  {}

  Index rows() const { return (size_N + size_M) * size_Q + size_M; }
  Index cols() const { return (2*size_Q + 1) * size_M + 1; }  

  template<typename Rhs, typename Dest>
  void applyThisOnTheLeft(Dest& r, const Rhs& x) const {
    assert(x.size() == this->cols() && "Incorrect size of input vec");
    assert(r.size() == this->rows() && "Incorrect size of output vec");
   
    auto size_X = size_M * size_Q;
    auto vec_X = x.segment(1, size_X);
    const auto& mat_X = vec_X.reshaped<Eigen::ColMajor>(size_M, size_Q);
    const auto& vec_R = x.segment(size_X + 1, size_X); // Slack matrix for X <= 1.
    const auto& vec_S = x.segment(size_X * 2 + 1, size_M);

    auto size_O = size_N * size_Q;
    r.segment(0, size_O) = ((*pmat_M) * mat_X).reshape<Eigen::ColMajor>(size_N * size_Q, 1);
    r.segment(size_O, size_X) = vec_X + vec_R;
    r.segment(size_O + size_X, size_M) = 
      x(0) * (*pvec_C) + mat_X * (*pvec_VQ) + vec_S;
  }

  template<typename Rhs, typename Dest>
  void applyThisOnTheRight(Dest& r, const Rhs& y) const { 
    assert(r.size() == this->cols() && "Incorrect size of output vec");
    assert(y.size() == this->rows() && "Incorrect size of input vec");

    auto size_Y = size_N * size_Q;
    auto size_O = size_M * size_Q;
    const auto& vec_Y = y.segment(0, size_Y);
    const auto& mat_Y = vec_Y.reshaped<Eigen::ColMajor>(size_M, size_Q);
    const auto& vec_L = y.segment(size_Y + size_O, size_M);

    r(0) = (*pvec_C).dot(y.segment(size_O + size_Y, size_M));
    r.segment(1, size_O) =(*pmat_M).transpose() * mat_Y + y.segment(size_Y, size_O);
    for (Index i = 0; i < size_Q; ++i) {
      r.segment(1 + i * size_M, size_M) += (*pvec_VQ)(i) * vec_L;
    }
    r.segment(1 + size_O, size_O + size_M) = y.segment(size_Y, size_O + size_M);
  }
};

class PDIPMMatrix;

namespace Eigen {
  namespace internal {
    template<>
    struct traits<PDIPMMatrix> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
    {};
  }
}


class PDIPMMatrix: public Eigen::EigenBase<PDIPMMatrix> {
public:
  using Scalar = double;
  using RealScalar = double;
  using StorageIndex = Eigen::Index;

  // Specify dynamic size (required by Eigen solvers)
  enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic,
      IsRowMajor = false
  };

  PDIPMMatrix(const PrimalMatrix* pmat_A,
              Eigen::VectorXd* pvec_V,
              Eigen::VectorXd* pvec_L )
    : pmat_A{ pmat_A }, 
      pvec_V{ pvec_V }, 
      pvec_L{ pvec_L },
      row_A{ pmat_A->rows() },
      len_V{ pmat_A->cols() }
  { 
    assert(pvec_L->size() == pvec_L->size() && "Vector V/L size inconsist");
    assert(pmat_A->cols() == pvec_L->size() && "Vector/Matrix size inconsist");
  }

  Eigen::Index rows() const { return row_A + 2 * len_V; }
  Eigen::Index cols() const { return row_A + 2 * len_V; }

  // --- Matrix-vector product ---
  template<typename Rhs>
  Eigen::Product<PDIPMMatrix,Rhs,Eigen::AliasFreeProduct> 
  operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<PDIPMMatrix, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // --- Multiplication Implementation ---
  template<typename Rhs, typename Dest>
  void applyThisOnTheLeft(Dest& r, const Rhs& x) const {
    assert(x.size() == this->cols() && "Input vector x has incorrect size for perform_op");
    assert(r.size() == this->rows() && "Output vector r has incorrect size for perform_op");

    const auto& dvec_V = x.segment(0, len_V);
    const auto& dvec_W = x.segment(len_V, row_A);
    const auto& dvec_L = x.segment(len_V + row_A, len_V);

    pmat_A->applyThisOnTheLeft(y.segment(0, row_A), dvec_V);
    pmat_A->applyThisOnTheRight(y.segment(row_A, len_V), dvec_W);
    r.segment(row_A, len_V) += dvec_L;

    r.segment(row_A + len_V, len_V) = 
      pvec_L->cwiseProduct(dvec_V) + pvec_X->cwiseProduct(dvec_L); 
  }

  template<typename Rhs, typename Dest>
  void applyThisOnTheRight(Dest& r, const Rhs& y) const {
    assert(y.size() == this->rows() && "Input vector y has incorrect size for perform_op");
    assert(r.size() == this->cols() && "Output vector r has incorrect size for perform_op");

    const auto& dvec_U = y.segment(0, row_A);
    const auto& dvec_M = y.segment(row_A, len_V);
    const auto& dvec_D = y.segment(len_V + row_A, len_V);
 
    pmat_A->applyThisOnTheRight(r.segment(0, len_V), dvec_U);
    pmat_A->applyThisOnTheLeft(r.segment(len_V, row_A), dvec_M);
    r.segment(0, len_V) += pvec_L->cwiseProduct(dvec_D);
    r.segment(len_V + row_A, len_V) += dvec_M + pvec_V->cwiseProduct(dvec_D);
  }

private:
  const PrimalMatrix* pmat_A;
  Eigen::VectorXd* pvec_V;
  Eigen::VectorXd* pvec_L;
  const Index row_A;
  const Index len_V; // Equals to col_A. 
};

// --- Specialization for Eigen's Product mechanism ---
// This tells Eigen how to evaluate the product PDIPMMatrix * DenseVectorType
namespace Eigen {
namespace internal {

template<typename Rhs>
struct generic_product_impl<PDIPMMatrix, Rhs, SparseShape, DenseShape> // MatrixType * DenseVectorType
 : generic_product_impl_base<PDIPMMatrix, Rhs, generic_product_impl<PDIPMMatrix, Rhs> >
{
  using Scalar = typename Product<PDIPMMatrix, Rhs>::Scalar;

  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const PDIPMMatrix& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
    assert(alpha==Scalar(1) && "scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);
    
    Eigen::VectorXd temp_res(lhs.rows());
    lhs.applyThisOnTheLeft(temp_res, rhs);
    lhs.applyThisOnTheRight(dst, temp_res);
  }
};

} // namespace internal
} // namespace Eigen