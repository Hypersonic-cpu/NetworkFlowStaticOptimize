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
    int from, to;
    double caps, bkg;
    std::istringstream ss(line);
    ss >> from >> to ;
    edges.emplace_back(Tri{ cnt, from-1, 1.0});
    edges.emplace_back(Tri{ cnt, to-1, -1.0});
    if (is_OD) {
      ss >> caps;
      cap.emplace_back(caps);
    } else {
      ss >> caps >> bkg;
      cap.emplace_back(caps - bkg);
    }
    cnt++;
  }
  
  assert(cnt == EdgeNum);
  return std::make_pair(edges, cap);
}

int main()
{
  using Eigen::VectorXd;
  using std::cout, std::endl;

  const int Node = 15;
  const int Edge = 32;
  const int ODmd = 10;

  auto [edgesTri, edgesCap] = ReadEdge("../DataTest/Edges.ssv", Edge, false);
  auto [odTri, odCap] = ReadEdge("../DataTest/OD_Demand.ssv", ODmd, true);
  Eigen::SparseMatrix<double> EdgesMat (Edge, Node);
  EdgesMat.setFromTriplets(edgesTri.begin(), edgesTri.end());

  Eigen::SparseMatrix<double> ODMat (ODmd, Node);
  ODMat.setFromTriplets(odTri.begin(), odTri.end());

  Eigen::VectorXd EdgesCap = Eigen::VectorXd::Map(edgesCap.data(), edgesCap.size()).eval();
  Eigen::VectorXd ODCap = Eigen::VectorXd::Map(odCap.data(), odCap.size()).eval();

  cout << std::setw(4) << "Cap"; cout << "|";
  for (auto j = 0; j < Node; ++j)
    cout << std::setw(4) << j;
  cout << endl << endl;

  for (auto i = 0; i < Edge; ++i) {
    cout << std::setw(4) << edgesCap.at(i);
    cout << "|";
    for (auto j = 0; j < Node; ++j)
      cout << std::setw(4) << EdgesMat.coeff(i, j);
    cout << endl;
  }
  cout << endl << endl;
  EdgesMat = EdgesMat.transpose().eval();

  cout << std::setw(4) << "OD"; cout << "|";
  for (auto j = 0; j < Node; ++j)
    cout << std::setw(4) << j;
  cout << endl << endl;

  for (auto i = 0; i < ODmd; ++i) {
    cout << std::setw(4) << odCap.at(i);
    cout << "|";
    for (auto j = 0; j < Node; ++j)
      cout << std::setw(4) << ODMat.coeff(i, j);
    cout << endl;
  }
  cout << endl;

  PrimalMatrix matrix_A { ODmd, Node, Edge, &EdgesMat, &EdgesCap, &ODCap };
  VectorXd primalV (matrix_A.cols());
  VectorXd dualY   (matrix_A.rows());
  VectorXd slackS  (matrix_A.cols());
  PDIPMSubMatrix matrix_Sub { &matrix_A, &primalV, &slackS };
  VectorXd costC = VectorXd::Zero(matrix_A.cols());
  { costC(0) = 1; }
  VectorXd rhsB  = VectorXd::Zero(matrix_A.rows());

  cout << "Dimension Check: " << endl;
  cout << "Q " << ODmd << "\tM " << Edge << "\tN " << Node << endl;
  cout << "MatA Size " << matrix_A.rows() << " x " << matrix_A.cols() << "\tShould be " 
       << (Edge + Node) * ODmd + Edge << " x " << 1 + (2 * ODmd + 1) * Edge << endl;
  cout << "MatSub    " << matrix_Sub.rows() << " x " << matrix_Sub.cols() << "\tShould be " 
       << matrix_A.rows() << " square" << endl;
  
  InteriorPointParams p;
  PrimalDualInteriorPoint ipm { &matrix_Sub, &rhsB, &costC, p };
  cout << dualY << endl;

  auto iter = ipm.SolveInPlace(primalV, dualY);
  cout << "TOTAL ITER " << iter << endl;
}