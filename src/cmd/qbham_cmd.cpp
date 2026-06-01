/****************************************************************************
  PackageName  [ hamiltonian ]
  Synopsis     [ Define hamiltonian commands ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include "./qbham_cmd.hpp"

#include <string>

#include "argparse/arg_parser.hpp"
#include "cli/cli.hpp"
#include "cmd/device_mgr.hpp"
#include "cmd/qbham/ferm_to_qubit.hpp"
#include "cmd/qbham/read.hpp"
#include "cmd/qbham/sort.hpp"
#include "cmd/qbham/trotterize.hpp"
#include "cmd/qbham/write.hpp"
#include "cmd/qbham_mgr.hpp"
#include "cmd/tableau_mgr.hpp"
#include "hamiltonian/qubit_hamiltonian.hpp"
#include "util/data_structure_manager_common_cmd.hpp"

using namespace dvlab::argparse;

namespace qsyn::hamiltonian {

dvlab::Command qbham_print_cmd(QubitHamiltonianMgr const& qbham_mgr) {
    return dvlab::Command{
        "print",
        [](ArgumentParser& parser) {
            parser.description("Print the hamiltonian");

            parser.add_argument<bool>("-v", "--verbose")
                .action(store_true)
                .help("display each term of the hamiltonian");
        },
        [&](ArgumentParser const& parser) {
            if (!dvlab::utils::mgr_has_data(qbham_mgr)) {
                return dvlab::CmdExecResult::error;
            }

            auto const* qbham = qbham_mgr.get();
            size_t tpw        = 0;

            for (auto const& term : *qbham) {
                for (size_t i = 0; i < term.n_qubits(); ++i) {
                    if (!term.is_i(i)) {
                        tpw++;
                    }
                }
            }

            fmt::println("Qubit Hamiltonian ({} qubits, {} terms, weight = {})",
                         qbham_mgr.get()->n_qubits(),
                         qbham_mgr.get()->n_terms(),
                         tpw);
            if (parser.parsed("--verbose")) {
                fmt::println("{}", qbham_mgr.get()->to_string());
            }

            return dvlab::CmdExecResult::done;
        }};
}

dvlab::Command qbham_cmd(device::DeviceMgr& device_mgr, QubitHamiltonianMgr& qbham_mgr, tableau::TableauMgr& tableau_mgr) {
    (void)device_mgr;  // reserved for future qbham commands that need a device
    auto cmd = dvlab::utils::mgr_root_cmd(qbham_mgr);

    cmd.add_subcommand("qbham-cmd-group", dvlab::utils::mgr_list_cmd(qbham_mgr));
    // cmd.add_subcommand("qbham-cmd-group", qbham_new_cmd(qbham_mgr));
    cmd.add_subcommand("qbham-cmd-group", dvlab::utils::mgr_delete_cmd(qbham_mgr));
    cmd.add_subcommand("qbham-cmd-group", dvlab::utils::mgr_checkout_cmd(qbham_mgr));
    cmd.add_subcommand("qbham-cmd-group", dvlab::utils::mgr_copy_cmd(qbham_mgr));
    cmd.add_subcommand("qbham-cmd-group", qbham_jw_cmd(qbham_mgr));
    cmd.add_subcommand("qbham-cmd-group", qbham_ternary_tree_cmd(qbham_mgr));
    cmd.add_subcommand("qbham-cmd-group", qbham_read_cmd(qbham_mgr));
    cmd.add_subcommand("qbham-cmd-group", qbham_write_cmd(qbham_mgr));
    cmd.add_subcommand("qbham-cmd-group", qbham_print_cmd(qbham_mgr));
    cmd.add_subcommand("qbham-cmd-group", qbham_trotterize_cmd(qbham_mgr, tableau_mgr));
    cmd.add_subcommand("qbham-cmd-group", qbham_sort_cmd(qbham_mgr));

    return cmd;
}

bool add_qbham_cmds(dvlab::CommandLineInterface& cli, device::DeviceMgr& device_mgr, QubitHamiltonianMgr& qbham_mgr, tableau::TableauMgr& tableau_mgr) {
    if (!cli.add_command(qbham_cmd(device_mgr, qbham_mgr, tableau_mgr))) {
        spdlog::error("Registering \"qbham\" commands fails... exiting");
        return false;
    }

    return true;
}

}  // namespace qsyn::hamiltonian
