/*
  Unit tests for QASM import (from_qasm).
*/

#include "qcir/qcir_io.hpp"
#include "util/phase.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>

using namespace qsyn::qcir;

namespace {

std::filesystem::path write_temp_qasm(std::string const& contents) {
    auto const path = std::filesystem::temp_directory_path() / "qsyn_from_qasm_test.qasm";
    std::ofstream out(path);
    REQUIRE(out);
    out << contents;
    return path;
}

}  // namespace

TEST_CASE("from_qasm rejects unsupported gates", "[from_qasm]") {
    auto const path = write_temp_qasm(R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rzz(pi/4) q[0], q[1];
)");
    REQUIRE_FALSE(from_qasm(path));
}

TEST_CASE("from_qasm accepts native basis gates", "[from_qasm]") {
    auto const path = write_temp_qasm(R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rz(-pi/188) q[0];
cx q[0], q[1];
)");
    auto qcir = from_qasm(path);
    REQUIRE(qcir.has_value());
    REQUIRE(qcir->get_num_gates() == 2);
}

TEST_CASE("from_qasm phase_eps controls float-to-rational approximation", "[from_qasm]") {
    auto const path = write_temp_qasm(R"(OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rz(1.5540857275736744) q[0];
)");
    REQUIRE(from_qasm(path, 1e-8).has_value());

    auto const phase = dvlab::Phase::from_string<double>("1.5540857275736744", 1e-8);
    REQUIRE(phase.has_value());
    REQUIRE(phase->numerator() == 93);
    REQUIRE(phase->denominator() == 188);
}
