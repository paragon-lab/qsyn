/**
  PackageName  [ hamiltonian ]
  Synopsis     [ Define command to write hamiltonian to file ]
  Author       [ Design Verification Lab ]
*/

#pragma once

#include "cli/cli.hpp"
#include "cmd/qbham_mgr.hpp"

namespace qsyn::hamiltonian {

dvlab::Command qbham_write_cmd(QubitHamiltonianMgr const& qbham_mgr);

}  // namespace qsyn::hamiltonian
