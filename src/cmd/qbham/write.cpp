/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Implement command to write hamiltonian to file ]
  Author       [ Design Verification Lab ]
*/

#include "./write.hpp"

#include <filesystem>

#include "argparse/arg_type.hpp"
#include "hamiltonian/qubit_hamiltonian.hpp"
#include "util/data_structure_manager_common_cmd.hpp"

using namespace dvlab::argparse;
using dvlab::CmdExecResult;
using dvlab::Command;

namespace qsyn::hamiltonian {

dvlab::Command qbham_write_cmd(QubitHamiltonianMgr const& qbham_mgr) {
    return Command(
        "write",
        [](ArgumentParser& parser) {
            parser.description("Write the focused qubit Hamiltonian to a text file");

            parser.add_argument<std::string>("filepath")
                .constraint(path_writable)
                .help(
                    "The path to the output file. "
                    "File format: 'Coefficient PauliString' per line, matching `qbham read`.");
        },
        [&](ArgumentParser const& parser) {
            if (!dvlab::utils::mgr_has_data(qbham_mgr)) {
                return CmdExecResult::error;
            }

            auto const filepath = std::filesystem::path(parser.get<std::string>("filepath"));
            if (!write_qubit_hamiltonian(*qbham_mgr.get(), filepath)) {
                return CmdExecResult::error;
            }

            return CmdExecResult::done;
        });
}

}  // namespace qsyn::hamiltonian
