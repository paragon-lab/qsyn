/*
  PackageName  [ hamiltonian ]
  Synopsis     [ Define qubit Hamiltonian term class ]
  Author       [ Mu-Te Lau (joshmtlau) ]
*/

#pragma once

#include <fmt/format.h>

#include <complex>
#include <filesystem>
#include <optional>
#include <string_view>

#include "tableau/pauli_product_trait.hpp"

namespace qsyn {

namespace hamiltonian {

template <typename CoeffT>
class PauliTermInterface
    : public qsyn::tableau::PauliProductTrait<PauliTermInterface<CoeffT>> {
public:
    using Pauli        = qsyn::tableau::Pauli;
    using PauliProduct = qsyn::tableau::PauliProduct;
    PauliTermInterface(
        std::initializer_list<Pauli> const& pauli_list,
        CoeffT coeff)
        : _pauli_product(pauli_list, false), _coeff(coeff) {
        _normalize();
    }

    PauliTermInterface(std::string_view pauli_str, CoeffT coeff)
        : _pauli_product(pauli_str), _coeff(coeff) {
        _normalize();
    }

    PauliTermInterface(
        PauliProduct const& pauli_product, CoeffT coeff)
        : _pauli_product(pauli_product), _coeff(coeff) {
        _normalize();
    }

    template <std::input_iterator I, std::sentinel_for<I> S>
    requires std::same_as<std::iter_value_t<I>, Pauli>
    PauliTermInterface(I first, S last, CoeffT coeff)
        : _pauli_product(first, last, false), _coeff(coeff) {
        _normalize();
    }

    template <std::ranges::range R>
    requires std::same_as<std::ranges::range_value_t<R>, Pauli>
    PauliTermInterface(R const& r, CoeffT coeff)
        : _pauli_product(std::ranges::begin(r), std::ranges::end(r), false),
          _coeff(coeff) {
        _normalize();
    }

    size_t n_qubits() const { return _pauli_product.n_qubits(); }
    Pauli get_pauli_type(size_t i) const { return _pauli_product.get_pauli_type(i); }

    bool is_i(size_t i) const { return _pauli_product.is_i(i); }
    bool is_x(size_t i) const { return _pauli_product.is_x(i); }
    bool is_y(size_t i) const { return _pauli_product.is_y(i); }
    bool is_z(size_t i) const { return _pauli_product.is_z(i); }

    PauliProduct const& pauli_product() const { return _pauli_product; }
    PauliProduct& pauli_product() { return _pauli_product; }
    CoeffT coeff() const { return _coeff; }
    CoeffT& coeff() { return _coeff; }

    bool operator==(PauliTermInterface const& rhs) const {
        return _pauli_product == rhs._pauli_product && _coeff == rhs._coeff;
    }
    bool operator!=(PauliTermInterface const& rhs) const { return !(*this == rhs); }

    std::string to_string(char signedness = '-') const {
        return fmt::format("{} * {}", _coeff, _pauli_product.to_string(signedness));
    }
    std::string to_bit_string() const {
        return fmt::format(
            "{}  {}",
            _pauli_product.to_bit_string().substr(0, 2 * n_qubits() + 1), _coeff);
    }

    PauliTermInterface& h(size_t qubit) noexcept override {
        _pauli_product.h(qubit);
        _normalize();
        return *this;
    }
    PauliTermInterface& s(size_t qubit) noexcept override {
        _pauli_product.s(qubit);
        _normalize();
        return *this;
    }
    PauliTermInterface& cx(size_t control, size_t target) noexcept override {
        _pauli_product.cx(control, target);
        _normalize();
        return *this;
    }

    bool is_commutative(PauliTermInterface const& rhs) const {
        return _pauli_product.is_commutative(rhs._pauli_product);
    }

    bool is_diagonal() const { return _pauli_product.is_diagonal(); }

protected:
    qsyn::tableau::PauliProduct _pauli_product;
    CoeffT _coeff;

    void _normalize() {
        if (_pauli_product.is_neg()) {
            _pauli_product.negate();
            _coeff *= -1;
        }
    }
};

class HermitianPauliTerm : public PauliTermInterface<double> {
public:
    using PauliTermInterface::PauliTermInterface;
};
class ComplexPauliTerm : public PauliTermInterface<std::complex<double>> {
public:
    using PauliTermInterface::PauliTermInterface;
    // NOTE: the multiplication is only closed for ComplexPauliTerm
    // This is why we don't define it in the PauliTermInterface base class
    ComplexPauliTerm& operator*=(ComplexPauliTerm const& rhs);
    friend ComplexPauliTerm operator*(
        ComplexPauliTerm lhs, ComplexPauliTerm const& rhs) {
        lhs *= rhs;
        return lhs;
    }
};

ComplexPauliTerm to_complex_pauli_term(HermitianPauliTerm const& term);

/**
 * Convert a complex Pauli term to a Hermitian Pauli term.
 * This function ignores the imaginary part of the complex coefficient.
 *
 * @param term The complex Pauli term to convert.
 * @return The Hermitian Pauli term.
 */
HermitianPauliTerm to_hermitian_pauli_term(ComplexPauliTerm const& term);

template <typename CoeffT>
inline bool is_commutative(
    PauliTermInterface<CoeffT> const& lhs, PauliTermInterface<CoeffT> const& rhs) {
    return lhs.is_commutative(rhs);
}

class QubitHamiltonian
    : public qsyn::tableau::PauliProductTrait<QubitHamiltonian> {
public:
    QubitHamiltonian(size_t n_qubits);
    QubitHamiltonian(std::initializer_list<HermitianPauliTerm> const& terms);

    size_t n_qubits() const { return _terms.begin()->n_qubits(); }

    QubitHamiltonian& h(size_t qubit) noexcept override;
    QubitHamiltonian& s(size_t qubit) noexcept override;
    QubitHamiltonian& cx(size_t control, size_t target) noexcept override;

    QubitHamiltonian& add_term(HermitianPauliTerm const& term);
    template <std::input_iterator I, std::sentinel_for<I> S>
    QubitHamiltonian& add_terms(I first, S last) {
        for (auto it = first; it != last; ++it) {
            add_term(*it);
        }
        return *this;
    }
    template <std::ranges::range R>
    QubitHamiltonian& add_terms(R const& r) {
        return add_terms(std::ranges::begin(r), std::ranges::end(r));
    }

    auto begin() const {
        return _terms.begin();
    }
    auto end() const {
        return _terms.end();
    }
    auto begin() {
        return _terms.begin();
    }
    auto end() {
        return _terms.end();
    }

    auto n_terms() const {
        return _terms.size();
    }

    std::string to_string() const;

private:
    std::vector<HermitianPauliTerm> _terms;
    size_t _n_qubits;
};

/**
 * Check if all terms in the Hamiltonian are commutative.
 *
 * @param hamiltonian The Hamiltonian to check.
 * @return True if all terms are commutative, false otherwise.
 */
bool is_all_commutative(QubitHamiltonian const& hamilt);

}  // namespace hamiltonian
}  // namespace qsyn

std::optional<qsyn::hamiltonian::QubitHamiltonian> read_qubit_hamiltonian(
    std::filesystem::path const& filepath);

bool write_qubit_hamiltonian(
    qsyn::hamiltonian::QubitHamiltonian const& hamilt,
    std::filesystem::path const& filepath);

template <>
struct fmt::formatter<std::complex<double>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
    template <typename FormatContext>
    auto format(std::complex<double> const& c, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "({}, {})", c.real(), c.imag());
    }
};
