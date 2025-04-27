#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

#include "CompressedSpMV.hh"

template<typename TMat>
struct InteriorPointParams {
  int MaxIteration = 1000;
  TMat::Scalar GradTolerance = 1e-5;
  TMat::Scalar EqnTolerance = 1e-6;
  TMat::Scalar CentralPathTau = 0.5;
};

/**
 * @attention This class does NOT store any matrices. 
 * Matrices are passed by pointer-to-ColVec or Matrix.
 */
class PrimalDualInteriorPoint {
  using AMat = PrimalMatrix;
  using EqnMat = PDIPMSubMatrix;
  using Scalar = AMat::Scalar;
  using ColVec = Eigen::VectorXd;
  using RowVec = Eigen::RowVectorXd;
  using Index = AMat::StorageIndex;
public:
  PrimalDualInteriorPoint(EqnMat* mat_Eqn_ptr, 
                          const ColVec* vec_B_ptr, 
                          const ColVec* vec_C_ptr, 
                          InteriorPointParams<AMat> params)
  : pmat_A{ mat_Eqn_ptr->rawPrimalMatrix() },
    pmat_Eqn{ mat_Eqn_ptr },
    pvec_B{ vec_B_ptr },
    pvec_C{ vec_C_ptr },
    rhs_dim{ mat_Eqn_ptr->rawPrimalMatrix()->rows() },
    prob_dim{ mat_Eqn_ptr->rawPrimalMatrix()->cols() },
    params{ params }
  {
    assert(vec_B_ptr->size() == rhs_dim
           && "RHS/Matrix have inconsist sizes"); 
    assert(vec_B_ptr->size() == prob_dim
           && "RHS/Matrix have inconsist sizes"); 
  }

  void SolveInPlace(ColVec& var, ColVec& dual, 
                    Scalar overall_tol=1e-5, Scalar eqn_tol=1e-6) {
    assert(var.size() == prob_dim && "Var/Matrix have inconsist sizes");
    assert(dual.size() == rhs_dim && "Dual/Matrix have inconsist sizes");

    ColVec slack = *pvec_C - pmat_A->transpose().operator*(dual);

    ColVec d_primal(prob_dim);
    ColVec d_dual(prob_dim);
    int iteration = 0;
    do {
      ColVec r_primal = (*pvec_B) - (*pmat_A) * var;
      
      

    } while (++iteration <= params.MaxIteration
             && d_primal.squaredNorm() > params.GradTolerance );
  }
  
  Index fullDim() const { return 2*prob_dim + rhs_dim; }
  Index probDim() const { return prob_dim; }
  Index rhsDim()  const { return rhs_dim; }

private:
  InteriorPointParams<AMat> params;
  const Index prob_dim;
  const Index rhs_dim;
  const AMat* pmat_A;
  const ColVec* pvec_B;
  const ColVec* pvec_C;
  EqnMat* pmat_Eqn; 
};
