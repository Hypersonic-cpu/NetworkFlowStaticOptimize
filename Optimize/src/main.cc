#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

#include "CompressedSpMV.hh"
#include "PrimalDualInteriorPoint.hh"

using Tri = Eigen::Triplet<double>;

std::pair<
  std::vector<Tri>,
  std::vector<double> >
ReadEdge(std::string fileName, const int EdgeNum, bool is_OD) {
  std::vector<Tri> edges {};
  std::vector<double> cap {};
  edges.reserve(EdgeNum << 1);
  cap.reserve(EdgeNum); 

  std::ifstream file(fileName);
  std::string line;

  // Skip the header row
  std::getline(file, line);

  // Process each data row
  int cnt = 0;
  while (std::getline(file, line)) {
    int from, to, no;
    double caps, bkg;
    std::istringstream ss(line);
    ss >> no >> from >> to ;
    edges.emplace_back(Tri{ cnt, from, 1.0});
    edges.emplace_back(Tri{ cnt, to, -1.0});
    if (is_OD) {
      ss >> caps;
      cap.emplace_back(caps);
    } else {
      ss >> caps >> bkg;
      cap.emplace_back(caps - bkg);
    }
    cnt++;
  }
  file.close();  
  assert(cnt == EdgeNum);
  return std::make_pair(edges, cap);
}

Eigen::VectorXd ReadInit(int Q, int M) {
  Eigen::VectorXd ret(2*M*Q + M + 1);
  ret(0) = 0.96875;
  std::ifstream file1("../Data/x_v.txt");
  for (auto i = 0; i < M*Q; ++i) {
    file1 >> ret(1+i);
  }
  std::ifstream file2("../Data/r_v.txt");
  for (auto i = 0; i < M*Q; ++i) {
    file2 >> ret(1+M*Q + i);
  }
  std::ifstream file3("../Data/s_del.txt");
  for (auto i = 0; i < M; ++i) {
    file3 >> ret(1+2*M*Q + i);
  }
  file1.close(); file2.close(); file3.close();
  return ret;
}

int main()
{
  using Eigen::VectorXd;
  using std::cout, std::endl;
 
  const int Node = 147;
  const int Edge = 556;
  const int ODmd = 500;

  auto [edgesTri, edgesCap] = ReadEdge("../Data/EdgeCapacity.ssv", Edge, false);
  auto [odTri, odCap] = ReadEdge("../Data/OD_Demand.ssv", ODmd, true);
  Eigen::SparseMatrix<double> EdgesMat (Edge, Node);
  EdgesMat.setFromTriplets(edgesTri.begin(), edgesTri.end());

  Eigen::SparseMatrix<double> ODMat (ODmd, Node);
  ODMat.setFromTriplets(odTri.begin(), odTri.end());

  Eigen::VectorXd EdgesCap = Eigen::VectorXd::Map(edgesCap.data(), edgesCap.size()).eval();
  Eigen::VectorXd ODCap = Eigen::VectorXd::Map(odCap.data(), odCap.size()).eval();

  // cout << std::setw(4) << "Cap"; cout << "|";
  // for (auto j = 0; j < Node; ++j)
  //   cout << std::setw(4) << j;
  // cout << endl << endl;

  // for (auto i = 0; i < Edge; ++i) {
  //   cout << std::setw(4) << edgesCap.at(i);
  //   cout << "|";
  //   for (auto j = 0; j < Node; ++j)
  //     cout << std::setw(4) << EdgesMat.coeff(i, j);
  //   cout << endl;
  // }
  // cout << endl << endl;
  EdgesMat = EdgesMat.transpose().eval(); 

  // cout << std::setw(4) << "OD"; cout << "|";
  // for (auto j = 0; j < Node; ++j)
  //   cout << std::setw(4) << j;
  // cout << endl << endl;

  // for (auto i = 0; i < ODmd; ++i) {
  //   cout << std::setw(4) << odCap.at(i);
  //   cout << "|";
  //   for (auto j = 0; j < Node; ++j)
  //     cout << std::setw(4) << ODMat.coeff(i, j);
  //   cout << endl;
  // }
  // cout << endl;

  PrimalMatrix matrix_A { ODmd, Node, Edge, &EdgesMat, &EdgesCap, &ODCap };
   
  VectorXd primalV = ReadInit(ODmd, Edge);
  primalV = primalV * 0.999 + VectorXd::Ones(primalV.size()) * 0.0001;  
  VectorXd dualY = VectorXd::Zero(matrix_A.rows());
  // VectorXd::Random(matrix_A.rows()).cwiseAbs() * 0.1 + 0.1 * VectorXd::Ones(matrix_A.rows());
  VectorXd slackS = primalV.cwiseInverse() * 1e-7;
  PDIPMSubMatrix matrix_Sub { matrix_A, &primalV, &slackS };
  VectorXd costC = VectorXd::Zero(matrix_A.cols());
  { costC(0) = 1; } 
  VectorXd rhsB  = VectorXd::Zero(matrix_A.rows());  
  {
    for (int k = 0; k < ODMat.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(ODMat, k); it; ++it) {
          int row = it.row();
          int col = it.col();
          double value = it.value();
          int index = col * ODMat.rows() + row;
          rhsB(index) = value;
          assert(index < Node * ODmd); 
      } 
    }
    rhsB.segment((Node)*ODmd, ODmd * Edge).noalias() = VectorXd::Ones(ODmd * Edge);
  } 


  cout << "Dimension Check: " << endl;
  cout << "Q " << ODmd << "\tM " << Edge << "\tN " << Node << endl;
  cout << "MatA Size " << matrix_A.rows() << " x " << matrix_A.cols() << "\tShould be " 
       << (Edge + Node) * ODmd + Edge << " x " << 1 + (2 * ODmd + 1) * Edge << endl;
  cout << "MatSub    " << matrix_Sub.rows() << " x " << matrix_Sub.cols() << "\tShould be " 
       << matrix_A.rows() << " square" << endl; 
 
  InteriorPointParams p;
  PrimalDualInteriorPoint ipm { &matrix_Sub, &rhsB, &costC, p };
  // cout << dualY << endl;   

  auto iter = ipm.SolveInPlace(primalV, dualY, slackS);
  cout << "TOTAL ITER " << iter << endl; 
}