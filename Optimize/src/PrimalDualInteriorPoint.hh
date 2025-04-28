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
  int MaxIteration = 10000;
  double GradTolerance = 1e-5;
  double EqnTolerance = 1e-6;
  double KKTCondSigma = 0.95;
  double CentralPathTau = 2e-4;
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

  int SolveInPlace(ColVec& var, ColVec& dual, ColVec& slack, 
                    Scalar overall_tol=1e-5) {
    using std::cout, std::endl;
    assert(var.size() == prob_dim && "Var/Matrix have inconsist sizes");
    assert(dual.size() == rhs_dim && "Dual/Matrix have inconsist sizes");

    cout << "Invoking SolveInPlace" << endl;
    cout << pvec_C->size() << " " << pmat_A->transpose().cols() << " " << dual.size() << endl;

    ColVec d_primal(prob_dim);
    ColVec d_dual(rhs_dim);
    ColVec d_slack(prob_dim);
    ColVec rhs_dual(rhs_dim);
    ColVec r_primal(prob_dim);
    ColVec r_dual(rhs_dim);
    ColVec r_comp(prob_dim);

    int iteration = 0;
    cout << "Invoking SolveInPlace" << endl;
    Scalar curr_mu = var.dot(slack) / prob_dim;
    Scalar alpha = params.CentralPathRho;

    do {
      cout << "Iter #" << iteration << "\t" << endl;
      r_primal.noalias() = (*pvec_B) - (*pmat_A) * var;
      r_dual.noalias()   = (*pvec_C) - slack - pmat_A->transpose().operator*(dual);
      r_comp.noalias()   = 
        params.KKTCondSigma * curr_mu * ColVec::Ones(prob_dim) - var.cwiseProduct(slack);

      rhs_dual = r_primal + pmat_A->operator*(
        (var.cwiseProduct(r_dual) - r_comp).cwiseQuotient(slack)
      );

      // cout << "primal " << var.transpose() << endl;
      // cout << "dual   " << dual.transpose() << endl;
      // cout << "slack " << slack.transpose() << endl;
      // cout << endl;
      // cout << "primal " << r_primal.transpose() << endl;
      // cout << "dual   " << pvec_C->transpose() - slack.transpose()-pmat_A->transpose().operator*(dual).transpose() << endl;
      // cout << "slack " << r_comp.transpose() << endl;

      Eigen::ConjugateGradient<EqnMat, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
      cg.setMaxIterations(2).compute(*pmat_Eqn);
      d_dual = cg.solve(rhs_dual);

      d_slack = r_dual - pmat_A->transpose().operator*(d_dual);
      d_primal = (r_comp - var.cwiseProduct(d_slack)).cwiseQuotient(slack);

      alpha = params.CentralPathRho;
      while (true) {
        ColVec nxt_primal = var + d_primal * alpha;
        ColVec nxt_slack = slack + d_slack * alpha;
        cout << curr_mu << " " << params.CentralPathTau * curr_mu << " " <<  alpha << " " <<(nxt_primal.cwiseProduct(nxt_slack)).minCoeff() << endl;
        int aaa;
        std::cin >> aaa;
        if (
          // ((nxt_primal.cwiseProduct(nxt_slack)).array() 
            // >= params.CentralPathTau * curr_mu).all()
            true
          && (nxt_primal.array() >= 0).all()
          && (nxt_slack.array() >= 0).all()
        ) { break; }
        alpha = alpha * params.CentralPathRho;
      }
      
      cout << "alpha " << alpha << "\t" << "delta " << d_primal.squaredNorm() << endl;
      var += alpha * d_primal;
      dual += alpha * d_dual;
      slack += alpha * d_slack;
      curr_mu = var.dot(slack) / prob_dim;
      cout << "Val " << var(0) << endl;
    } while (++iteration <= params.MaxIteration
             && curr_mu > params.GradTolerance );
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
