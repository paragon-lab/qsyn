/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Define Trotterization functions ]
  Author       [ Mu-Te Lau (joshmtlau) ]
    */

#include "trotterize.hpp"

#include "tableau/pauli_rotation.hpp"

namespace qsyn::hamiltonian {

namespace {

/**
 * trotterize a single step of a Hamiltonian composed of only commutative terms.
 * Basically, for each term $H_i$ in a Hamiltonian $H$, this function appends
 * Pauli rotations $U_i = \exp(-i H_i \Delta t)$ to the given tableau.
 *
 * @param tableau The tableau to append the Trotterization to.
 * @param hamiltonian The Hamiltonian to trotterize.
 * @param dt The time step.
 */
void append_trotterize_step(
    qsyn::tableau::PauliRotationTableau& prtabl,
    QubitHamiltonian const& hamilt,
    double dt) noexcept {
    using qsyn::tableau::PauliRotation;

    for (auto const& term : hamilt) {
        if (term.coeff() == 0.0) {
            continue;
        }
        if (term.pauli_product().is_identity()) {
            continue;
        }
        // PZGate implements exp(-i θ/2 Z) on the rotation qubit,
        // so evolution exp(-i c Δt Z) needs θ = 2|c|Δt
        // (c.f. Qiskit's PauliEvolutionGate).
        auto const phase = dvlab::Phase::to_phase(2 * term.coeff() * dt, 1e-8);
        prtabl.push_back(PauliRotation(term.pauli_product(), phase));
    }
}

}  // namespace

qsyn::tableau::PauliRotationTableau trotterize_single_step(
    QubitHamiltonian const& hamiltonian, double dt) noexcept {
    using qsyn::tableau::PauliRotationTableau;

    auto prtabl = PauliRotationTableau{};

    if (hamiltonian.n_terms() == 0 || dt == 0) {
        return prtabl;
    }

    prtabl.reserve(hamiltonian.n_terms());

    append_trotterize_step(prtabl, hamiltonian, dt);

    return prtabl;
}
/**
 * Trotterize a Hamiltonian for a given time and number of steps.
 * If the Hamiltonian is commutative, the resulting PauliRotationTableau is
 * exact. Otherwise, it implements the Hamiltonian up to an error of
 * order $O(t^2/\mathrm{n\_steps})$.
 *
 * @param hamiltonian The Hamiltonian to trotterize.
 * @param time The time to trotterize for.
 * @param n_steps The number of steps to trotterize for. setting this to 0
 *        will return an empty PauliRotationTableau.
 * @return The Trotterized PauliRotationTableau.
 */
qsyn::tableau::PauliRotationTableau trotterize(
    QubitHamiltonian const& hamiltonian, double time, size_t n_steps) noexcept {
    using dvlab::iterator::next;
    using qsyn::tableau::PauliRotationTableau;

    if (n_steps == 0) {
        return PauliRotationTableau{};
    }

    auto const all_commutative = is_all_commutative(hamiltonian);

    if (all_commutative) {
        n_steps = 1;
    }

    auto const dt = time / static_cast<double>(n_steps);

    auto prtabl = trotterize_single_step(hamiltonian, dt);

    if (!all_commutative) {
        auto const n_rotations_per_step = prtabl.size();
        prtabl.reserve(n_rotations_per_step * n_steps);

        // if the Hamiltonian is not commutative, repeat (n_steps - 1) time more
        // since all steps are the same, we can just copy the existing terms
        for (size_t i = 1; i < n_steps; ++i) {
            prtabl.insert(prtabl.end(),
                          prtabl.begin(), next(prtabl.begin(), n_rotations_per_step));
        }
    }

    return prtabl;
}

}  // namespace qsyn::hamiltonian
