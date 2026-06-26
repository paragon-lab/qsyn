/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Fermion-to-qubit mappings (reusable F2Q logic) ]
  Author       [ April Wang (april864) ]
*/

#pragma once

#include <memory>

#include "hamiltonian/fermionic_hamiltonian.hpp"
#include "hamiltonian/qubit_hamiltonian.hpp"
#include "tableau/stabilizer_tableau.hpp"
#include "ternarytree/ternary_tree.hpp"
#include "ternarytree/tt_mappings.hpp"

namespace qsyn::hamiltonian {

// Fermion-to-qubit mapping interface
class FermionToQubitMapping {
public:
    FermionToQubitMapping(std::size_t n_modes) : _n_modes(n_modes) {}
    virtual ~FermionToQubitMapping() = default;
    /**
     * @brief Map an annihilation or creation operator to a qubit Hamiltonian.
     * @param n_qubits The number of qubits.
     * @param p The index of the fermion.
     * @param is_creation Whether the operator is a creation (true) or
                          annihilation (false) operator.
     * @return The list of ComplexPauliTerms representing the mapped operator.
     */
    virtual std::vector<ComplexPauliTerm>
    map(std::size_t p, bool is_creation) const = 0;

    virtual std::unique_ptr<FermionToQubitMapping> clone() const = 0;

    /// Clifford ``C`` with ``Phi_mapping = C Phi_JW`` (identity for Jordan–Wigner).
    virtual tableau::StabilizerTableau to_clifford() const = 0;

    std::size_t n_modes() const { return _n_modes; }

protected:
    std::size_t _n_modes;
};

class JordanWignerMapping : public FermionToQubitMapping {
public:
    JordanWignerMapping(std::size_t n_modes)
        : FermionToQubitMapping(n_modes) {}
    ~JordanWignerMapping() override = default;
    std::vector<ComplexPauliTerm> map(
        std::size_t p, bool is_creation) const override;

    std::unique_ptr<FermionToQubitMapping> clone() const override;

    tableau::StabilizerTableau to_clifford() const override;
};

class TernaryTreeMapping : public FermionToQubitMapping {
public:
    TernaryTreeMapping(std::size_t n_modes)
        : FermionToQubitMapping(n_modes), _mapper(n_modes) {}
    explicit TernaryTreeMapping(TernaryTree tree)
        : FermionToQubitMapping(tree.num_qubits()), _mapper(std::move(tree)) {}
    TernaryTreeMapping(TernaryTreeMapping const& other)
        : FermionToQubitMapping(other), _mapper(other._mapper.tree()) {}
    TernaryTreeMapping(TernaryTreeMapping&&) noexcept = default;
    TernaryTreeMapping& operator=(TernaryTreeMapping const& other) {
        if (this != &other) {
            FermionToQubitMapping::operator=(other);
            _mapper = TTMapper(other._mapper.tree());
        }
        return *this;
    }
    TernaryTreeMapping& operator=(TernaryTreeMapping&&) noexcept = default;
    ~TernaryTreeMapping() override                               = default;

    std::vector<ComplexPauliTerm> map(std::size_t p, bool is_creation) const override {
        return _mapper.get_pauli_str(p, is_creation);
    }

    std::unique_ptr<FermionToQubitMapping> clone() const override {
        return std::make_unique<TernaryTreeMapping>(_mapper.tree());
    }

    tableau::StabilizerTableau to_clifford() const override;

    TernaryTree const& tree() const { return _mapper.tree(); }

    TTMapper const& tt_mapper() const { return _mapper; }

private:
    TTMapper _mapper;
};

QubitHamiltonian qubitize(
    FermionHamiltonian const& f_hamilt,
    FermionToQubitMapping const& mapping);

}  // namespace qsyn::hamiltonian
