#pragma once 

#include <iostream>
#include <vector>
#include <cassert>
#include <type_traits>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

class PDIPMSubMatrix;
class PrimalMatrix;

namespace Eigen {
  namespace internal {
    template<>
    struct traits<PDIPMSubMatrix> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
    {};
    template<>
    struct traits<PrimalMatrix> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
    {};
  }
}


struct PrimalMatrix : public Eigen::EigenBase<PrimalMatrix> {
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

  explicit
  PrimalMatrix(Index size_Q, Index size_N, Index size_M, 
               const Eigen::SparseMatrix<double>* mat_M_ptr,
               const Eigen::VectorXd* vec_C_ptr,
               const Eigen::VectorXd* vec_VQ_ptr)
              : size_Q{ size_Q }, size_M{ size_M }, size_N{ size_N },
                pmat_M{ mat_M_ptr }, 
                pvec_C{ vec_C_ptr }, 
                pvec_VQ{ vec_VQ_ptr },
                is_transposed{ false }
  {}

  Index rows() const { return (size_N + size_M) * size_Q + size_M; }
  Index cols() const { return (2*size_Q + 1) * size_M + 1; }  

  template<typename Rhs>
  Eigen::Product<PrimalMatrix, Rhs,Eigen::AliasFreeProduct>
  operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<PrimalMatrix, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

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

  PrimalMatrix transpose() const {
    auto ret = *this;
    ret.is_transposed = !this->is_transposed;
    return ret;
  }

  bool transposed() const { return this->is_transposed; }

private:
  const Eigen::SparseMatrix<double>* pmat_M;
  const Eigen::VectorXd* pvec_C;  // col vec
  const Eigen::VectorXd* pvec_VQ; // col vec
  const Index size_Q;
  const Index size_M;
  const Index size_N;
  bool is_transposed;
};

/**
 * @brief The matrix `AS^{-1}XA^T` for solving delta_y 
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

  PDIPMSubMatrix(const PrimalMatrix* pmat_A,
              Eigen::VectorXd* pvec_V,
              Eigen::VectorXd* pvec_L )
    : pmat_A{ pmat_A }, 
      pvec_V{ pvec_V }, 
      pvec_L{ pvec_L },
      row_A{ pmat_A->rows() },
      len_V{ pmat_A->cols() },
      is_transposed{ false }
  { 
    assert(pvec_L->size() == pvec_L->size() && "Vector V/L size inconsist");
    assert(pmat_A->cols() == pvec_L->size() && "Vector/Matrix size inconsist");
  }

  Eigen::Index rows() const { return row_A; }
  Eigen::Index cols() const { return row_A; }

  // --- Matrix-vector product ---
  template<typename Rhs>
  Eigen::Product<PDIPMSubMatrix,Rhs,Eigen::AliasFreeProduct> 
  operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<PDIPMSubMatrix, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // --- Multiplication Implementation ---
  template<typename Rhs, typename Dest>
  void applyThisOnTheLeft(Dest& r, const Rhs& x) const {
    assert(x.size() == this->cols() && "Input vector x has incorrect size for perform_op");
    assert(r.size() == this->rows() && "Output vector r has incorrect size for perform_op");

    Eigen::VectorXd temp_res(len_V);
    pmat_A->applyThisOnTheRight(temp_res, x);
    pmat_A->applyThisOnTheLeft(
      r, temp_res.cwiseProduct((pvec_V->array() / pvec_L->array()).matrix()));
  }

  template<typename Rhs, typename Dest>
  void applyThisOnTheRight(Dest& r, const Rhs& y) const {
    applyThisOnTheLeft(r, y);
  }

  const PrimalMatrix* rawPrimalMatrix() const { return pmat_A; }

  PDIPMSubMatrix transpose() const {
    assert(false && "Should not be transposed");
    auto ret = *this;
    ret.is_transposed = !this->is_transposed;
    return ret;
  }
  bool transposed() const { return this->is_transposed; }

private:
  const PrimalMatrix* pmat_A;
  Eigen::VectorXd* pvec_V;
  Eigen::VectorXd* pvec_L;
  const Index row_A;
  const Index len_V; // Equals to col_A. 
  bool is_transposed;
};

namespace Eigen {
namespace internal {
template<typename T>
struct ipm_matrix_class : std::false_type {};
template<>
struct ipm_matrix_class<PDIPMSubMatrix>: std::true_type {};
template<>
struct ipm_matrix_class<PrimalMatrix>: std::true_type {};

// Matrix * DenseVector
template<typename Lhs, typename Rhs>
struct generic_product_impl<Lhs, Rhs, SparseShape, DenseShape> 
 : generic_product_impl_base<Lhs, Rhs, generic_product_impl<Lhs, Rhs> >
{
  using Scalar = typename Product<Lhs, Rhs>::Scalar;

  template <typename Dest>
  typename std::enable_if<ipm_matrix_class<Lhs>::value>::type
  scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
    // assert(alpha==Scalar(1) && "scaling is not implemented");
    // EIGEN_ONLY_USED_FOR_DEBUG(alpha);

    if (lhs.transposed()) {
      lhs.applyThisOnTheRight(dst, rhs);
    } else {
      lhs.applyThisOnTheLeft(dst, rhs);
    }
    dst = (alpha * dst).eval();
  }
};
} // namespace internal
} // namespace Eigen