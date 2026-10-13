/****************************************************************************
  PackageName  [ tableau ]
  Synopsis     [ Define pauli rotation class ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#pragma once

#include <fmt/core.h>

#include <initializer_list>
#include <iterator>
#include <ranges>
#include <sul/dynamic_bitset.hpp>
#include <tl/zip.hpp>
#include <vector>

#include "tableau/pauli_product_trait.hpp"
#include "util/phase.hpp"

namespace qsyn {

namespace tableau {

class PauliRotation : public PauliProductTrait<PauliRotation> {
public:
    PauliRotation(std::initializer_list<Pauli> const& pauli_list, dvlab::Phase const& phase);
    PauliRotation(std::string_view pauli_str, dvlab::Phase const& phase);

    PauliRotation(PauliProduct const& pauli_product, dvlab::Phase const& phase);

    template <std::input_iterator I, std::sentinel_for<I> S>
    requires std::same_as<std::iter_value_t<I>, Pauli>
    PauliRotation(I first, S last, dvlab::Phase const& phase) : _pauli_product(first, last, false), _phase(phase) {
        _normalize();
    }

    template <std::ranges::range R>
    requires std::same_as<std::ranges::range_value_t<R>, Pauli>
    PauliRotation(R const& r, dvlab::Phase const& phase)
        : PauliRotation(std::ranges::begin(r), std::ranges::end(r), phase) {}

    size_t n_qubits() const { return _pauli_product.n_qubits(); }
    Pauli get_pauli_type(size_t i) const { return _pauli_product.get_pauli_type(i); }

    bool is_i(size_t i) const { return _pauli_product.is_i(i); }
    bool is_x(size_t i) const { return _pauli_product.is_x(i); }
    bool is_y(size_t i) const { return _pauli_product.is_y(i); }
    bool is_z(size_t i) const { return _pauli_product.is_z(i); }

    sul::dynamic_bitset<> non_identity_mask() const { return _pauli_product.non_identity_mask(); }
    size_t weight() const { return _pauli_product.weight(); }
    std::vector<size_t> non_identity_qubits() const { return _pauli_product.non_identity_qubits(); }

    PauliProduct const& pauli_product() const { return _pauli_product; }
    dvlab::Phase const& phase() const { return _phase; }
    dvlab::Phase& phase() { return _phase; }

    bool operator==(PauliRotation const& rhs) const {
        return _pauli_product == rhs._pauli_product && _phase == rhs._phase;
    }
    bool operator!=(PauliRotation const& rhs) const { return !(*this == rhs); }

    std::string to_string(char signedness = '-') const;
    std::string to_bit_string() const;

    PauliRotation& h(size_t qubit) noexcept override;
    PauliRotation& s(size_t qubit) noexcept override;
    PauliRotation& cx(size_t control, size_t target) noexcept override;

    bool is_commutative(PauliRotation const& rhs) const {
        return _pauli_product.is_commutative(rhs._pauli_product);
    }

    bool is_diagonal() const { return _pauli_product.is_diagonal(); }

private:
    PauliProduct _pauli_product;
    dvlab::Phase _phase;

    void _normalize() {
        if (_pauli_product.is_neg()) {
            _pauli_product.negate();
            _phase *= -1;
        }
    }
};

inline bool is_commutative(PauliRotation const& lhs, PauliRotation const& rhs) {
    return lhs.is_commutative(rhs);
}

std::pair<CliffordOperatorString, size_t> extract_clifford_operators(PauliRotation pauli_rotation);
using PauliRotationTableau = std::vector<PauliRotation>;
size_t matrix_rank(std::vector<PauliRotation> const& rotations);

}  // namespace tableau

}  // namespace qsyn

template <>
struct fmt::formatter<qsyn::tableau::PauliRotation> {
    char presentation = 'c';
    char signedness   = '-';

    constexpr auto parse(format_parse_context& ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == '+' || *it == '-' || *it == ' ')) signedness = *it++;
        if (it != end && (*it == 'c' || *it == 'b')) presentation = *it++;
        if (it != end && *it != '}') detail::throw_format_error("invalid format");
        return it;
    }

    template <typename FormatContext>
    auto format(qsyn::tableau::PauliRotation const& c, FormatContext& ctx) const {
        if (presentation == 'b') {
            return fmt::format_to(ctx.out(), "{}", c.to_bit_string());
        } else {
            return fmt::format_to(ctx.out(), "{}", c.to_string(signedness));
        }
    }
};
