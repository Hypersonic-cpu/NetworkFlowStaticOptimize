#pragma once 

#include <iostream>
#include <vector>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

class PDIPMSubMatrix;
class PrimalMatrix;
class PrimalMatrixSymmetric;

namespace Eigen {
  namespace internal {
    template<>
    struct traits<PDIPMSubMatrix> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
    {};
    template<>
    struct traits<PrimalMatrix> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
    {};
    template<>
    struct traits<PrimalMatrixSymmetric> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
    {};
  }
}

/**
 * @attention This class does NOT store values. And DO pass
 * this type by values but not pointer.
 */
class PrimalMatrix : public Eigen::EigenBase<PrimalMatrix> {
public:
  using Scalar = double;
  using RealScalar = double;
  using StorageIndex = Eigen::Index;

  enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic,
      IsRowMajor = false
  };

  using Index = StorageIndex;

  PrimalMatrix(Index size_Q, Index size_N, Index size_M, 
               const Eigen::SparseMatrix<double>* mat_M_ptr,
               const Eigen::VectorXd* vec_C_ptr,
               const Eigen::VectorXd* vec_VQ_ptr)
              : size_Q{ size_Q }, size_M{ size_M }, size_N{ size_N },
                tot_rows{ (size_N + size_M) * size_Q + size_M },
                tot_cols{ (2*size_Q + 1) * size_M + 1 },
                pmat_M{ mat_M_ptr }, 
                pvec_C{ vec_C_ptr }, 
                pvec_VQ{ vec_VQ_ptr },
                is_transposed{ false }
  {}

  PrimalMatrix(const PrimalMatrix& other) = default;

  Index rows() const { return !is_transposed ? tot_rows : tot_cols; }
  Index cols() const { return !is_transposed ? tot_cols : tot_rows; }  

  template<typename Rhs>
  Eigen::Product<PrimalMatrix, Rhs,Eigen::AliasFreeProduct>
  operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<PrimalMatrix, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  PrimalMatrix transpose() const {
    auto ret = *this;
    ret.is_transposed = !this->is_transposed;
    return ret;
  }

  bool transposed() const { return this->is_transposed; }

  /**
   * @brief Apply this matrix on the left of the vector.
   */
  template<typename Rhs, typename Dest>
  void MultiplyVector(Dest& r, const Rhs& v) const { 
    if (transposed()) {
      _r_multiply(r, v);
    } else {
      _l_multiply(r, v);
    }
  }

protected:
  template<typename Rhs, typename Dest>
  void _l_multiply(Dest& r, const Rhs& x) const {
    assert(x.size() == tot_cols && "Incorrect size of input vec");
    assert(r.size() == tot_rows && "Incorrect size of output vec");
   
    auto size_X = size_M * size_Q;
    auto vec_X = x.segment(1, size_X);
    const auto& mat_X = vec_X.template reshaped<Eigen::ColMajor>(size_M, size_Q);
    const auto& vec_R = x.segment(size_X + 1, size_X); // Slack matrix for X <= 1.
    const auto& vec_S = x.segment(size_X * 2 + 1, size_M);
    // auto mat_VQ = pvec_VQ->asDiagonal();

    auto size_O = size_N * size_Q;
    r.segment(0, size_O).noalias() = ((*pmat_M) * mat_X).template reshaped<Eigen::ColMajor>(size_N * size_Q, 1);
    r.segment(size_O, size_X) = vec_X + vec_R;
    r.segment(size_O + size_X, size_M) = 
      x(0) * (*pvec_C) + mat_X * (*pvec_VQ) + vec_S;
  }

  template<typename Rhs, typename Dest>
  void _r_multiply(Dest& r, const Rhs& y) const { 
    assert(r.size() == tot_cols && "Incorrect size of output vec");
    assert(y.size() == tot_rows && "Incorrect size of input vec");

    auto size_Y = size_N * size_Q;
    auto size_O = size_M * size_Q;
    const auto& vec_Y = y.segment(0, size_Y);
    const auto& mat_Y = vec_Y.template reshaped<Eigen::ColMajor>(size_N, size_Q);
    const auto& vec_L = y.segment(size_Y + size_O, size_M);
    // auto mat_VQ = pvec_VQ->asDiagonal();

    r(0) = (*pvec_C).dot(y.segment(size_O + size_Y, size_M));
    r.segment(1, size_O) = 
      (pmat_M->transpose() * mat_Y).template reshaped<Eigen::ColMajor>(size_O, 1) 
      + y.segment(size_Y, size_O);
    for (Index i = 0; i < size_Q; ++i) {
      r.segment(1 + i * size_M, size_M).noalias() += (*pvec_VQ)(i) * vec_L;
    }
    r.segment(1 + size_O, size_O + size_M).noalias() = y.segment(size_Y, size_O + size_M);
  }

private:
  const Eigen::SparseMatrix<double>* pmat_M;
  const Eigen::VectorXd* pvec_C;  // col vec
  const Eigen::VectorXd* pvec_VQ; // col vec
  const Index size_Q;
  const Index size_M;
  const Index size_N;
  const Index tot_rows;
  const Index tot_cols;
  bool is_transposed;
};

/**
 * @brief The matrix `AS^{-1}XA^T` for solving delta_dual 
 * in primal-dual IPM.
 */
class PDIPMSubMatrix: public Eigen::EigenBase<PDIPMSubMatrix> {
public:
  using Scalar = double;
  using RealScalar = double;
  using StorageIndex = Eigen::Index;

  enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic,
      IsRowMajor = false
  };

  PDIPMSubMatrix(const PrimalMatrix& mat_A,
              Eigen::VectorXd* pvec_V,
              Eigen::VectorXd* pvec_S )
    : mat_A{ mat_A }, 
      pvec_V{ pvec_V }, 
      pvec_S{ pvec_S },
      row_A{ mat_A.rows() },
      len_V{ mat_A.cols() } 
  { 
    assert(pvec_S->size() == pvec_V->size() && "Vector V/S size inconsist");
    assert(mat_A.cols() == pvec_S->size() && "Vector/Matrix size inconsist");
    assert(!mat_A.transposed());
  }

  Eigen::Index rows() const { return row_A; }
  Eigen::Index cols() const { return row_A; }

  // --- Matrix-vector product ---
  template<typename Rhs>
  Eigen::Product<PDIPMSubMatrix,Rhs,Eigen::AliasFreeProduct> 
  operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<PDIPMSubMatrix, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  const PrimalMatrix& rawPrimalMatrix() const { return mat_A; }

  constexpr bool transposed() const { return false; }

  /**
   * @brief Apply this matrix on the left of the vector.
   */
  template<typename Rhs, typename Dest>
  void MultiplyVector(Dest& r, const Rhs& v) const { 
    _l_multiply(r, v);
  }

protected:
  template<typename Rhs, typename Dest>
  void _l_multiply(Dest& r, const Rhs& x) const {
    assert(x.size() == this->cols() && "Input vector x has incorrect size for perform_op");
    assert(r.size() == this->rows() && "Output vector r has incorrect size for perform_op");

    r.noalias() = mat_A * ((mat_A.transpose() * x).cwiseProduct(*pvec_V).cwiseQuotient(*pvec_S));
  }

private:
  const PrimalMatrix mat_A;
  Eigen::VectorXd* pvec_V;
  Eigen::VectorXd* pvec_S;
  const Index row_A;
  const Index len_V; // Equals to col_A. 
};

namespace Eigen {
namespace internal {

// Matrix * DenseVector
template<typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

template<typename Lhs, typename Rhs>
requires IsAnyOf<Lhs, PrimalMatrix, PrimalMatrixSymmetric, PDIPMSubMatrix>
struct generic_product_impl<Lhs, Rhs, SparseShape, DenseShape> 
 : generic_product_impl_base<Lhs, Rhs, generic_product_impl<Lhs, Rhs> >
{
  using Scalar = typename Product<Lhs, Rhs>::Scalar;

  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
    assert(alpha==Scalar(1) && "scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);

    lhs.MultiplyVector(dst, rhs);
    // dst = (alpha * dst).eval();
  }
};
} // namespace internal
} // namespace Eigen