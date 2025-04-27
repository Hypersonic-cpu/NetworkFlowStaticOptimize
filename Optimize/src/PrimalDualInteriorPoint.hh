#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "CompressedSpMV.hh"

struct InteriorPointParams {
  int MaxIteration = 1000;
  double GradTolerance = 1e-5;
  double EqnTolerance = 1e-6;
  double KKTCondSigma = 0.95;
  double CentralPathTau = 0.01;
  double CentralPathRho = 0.3;
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
                          InteriorPointParams params)
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
    assert(vec_C_ptr->size() == prob_dim
           && "RHS/Matrix have inconsist sizes"); 
  }

  int SolveInPlace(ColVec& var, ColVec& dual, 
                    Scalar overall_tol=1e-5) {
    using std::cout, std::endl;
    assert(var.size() == prob_dim && "Var/Matrix have inconsist sizes");
    assert(dual.size() == rhs_dim && "Dual/Matrix have inconsist sizes");

    cout << "Invoking SolveInPlace" << endl;
    cout << pvec_C->size() << " " << pmat_A->transpose().cols() << " " << dual.size() << endl;
    ColVec slack = *pvec_C - pmat_A->transpose().operator*(dual);

    ColVec d_primal(prob_dim);
    ColVec d_dual(rhs_dim);
    ColVec d_slack(prob_dim);
    ColVec rhs_dual(rhs_dim);
    int iteration = 0;
    cout << "Invoking SolveInPlace" << endl;
    Scalar curr_mu = var.dot(slack) / prob_dim;

    do {
      cout << "Iter #" << iteration << "\t" << endl;
      ColVec r_primal = (*pvec_B) - (*pmat_A) * var;
      ColVec r_dual = (*pvec_C) - slack - pmat_A->transpose().operator*(dual);
      ColVec r_comp = 
        params.KKTCondSigma * curr_mu * ColVec::Ones(prob_dim) - var.cwiseProduct(slack);

      rhs_dual = r_primal + pmat_A->operator*(
        (var.cwiseProduct(r_dual) - r_comp).cwiseQuotient(slack)
      );

      Eigen::ConjugateGradient<EqnMat, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
      cg.compute(*pmat_Eqn);
      d_dual = cg.solve(rhs_dual);

      d_slack = r_dual - pmat_A->transpose().operator*(d_dual);
      d_primal = (r_comp - var.cwiseProduct(d_slack)).cwiseQuotient(slack);

      cout << d_dual.transpose() << endl;
      
      Scalar alpha = params.CentralPathRho;
      while (true) {
        int a;
        std::cin >> a;
        ColVec nxt_primal = var + d_primal * alpha;
        ColVec nxt_slack = slack + d_slack * alpha;
        cout << alpha << " | " << nxt_primal.transpose() << endl;
        cout << (nxt_primal.cwiseProduct(nxt_slack)).transpose() << endl;
        if (
          ((nxt_primal.cwiseProduct(nxt_slack)).array() 
            >= params.CentralPathTau * curr_mu).all()
          && (nxt_primal.array() >= 0).all()
          && (nxt_slack.array() >= 0).all()
        ) { break; }
        alpha = alpha * params.CentralPathRho;
      }
      
      cout << "alpha " << alpha << "\t" << "delta " << d_primal.squaredNorm() << endl;
      var += alpha * d_primal;
      dual += alpha * d_dual;
      slack += alpha * d_slack;

    } while (++iteration <= params.MaxIteration
             && d_primal.squaredNorm() > params.GradTolerance );
    return iteration;
  }
  
  Index fullDim() const { return 2*prob_dim + rhs_dim; }
  Index probDim() const { return prob_dim; }
  Index rhsDim()  const { return rhs_dim; }

private:
  InteriorPointParams params;
  const Index prob_dim;
  const Index rhs_dim;
  const AMat* pmat_A;
  const ColVec* pvec_B;
  const ColVec* pvec_C;
  EqnMat* pmat_Eqn; 
};
