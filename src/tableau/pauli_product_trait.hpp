/****************************************************************************
  PackageName  [ tableau ]
  Synopsis     [ Define pauli rotation class ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#pragma once

#include <fmt/core.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <ranges>
#include <sul/dynamic_bitset.hpp>
#include <tl/zip.hpp>
#include <vector>

#include "util/util.hpp"

namespace qsyn {

namespace tableau {

enum class Pauli : std::uint8_t {
    i,
    x,
    y,
    z
};

uint8_t power_of_i(Pauli a, Pauli b);

enum class CliffordOperatorType : std::uint8_t {
    h,
    s,
    cx,
    sdg,
    v,
    vdg,
    x,
    y,
    z,
    cz,
    swap,
    ecr,
};

std::optional<CliffordOperatorType> to_clifford_operator_type(std::string_view str) noexcept;
std::string to_string(CliffordOperatorType type);

using CliffordOperator       = std::pair<CliffordOperatorType, std::array<size_t, 2>>;
using CliffordOperatorString = std::vector<CliffordOperator>;

[[nodiscard]] inline CliffordOperatorType adjoint(CliffordOperatorType const& op) {
    using COT = CliffordOperatorType;
    switch (op) {
        case COT::s:
            return COT::sdg;
        case COT::sdg:
            return COT::s;
        case COT::v:
            return COT::vdg;
        case COT::vdg:
            return COT::v;
        default:
            return op;
    }
}

inline void adjoint_inplace(CliffordOperatorType& op) {
    op = adjoint(op);
}

[[nodiscard]] inline CliffordOperator adjoint(CliffordOperator const& op) {
    return {adjoint(op.first), op.second};
}

inline void adjoint_inplace(CliffordOperator& op) {
    op.first = adjoint(op.first);
}

inline void adjoint_inplace(CliffordOperatorString& ops) {
    std::ranges::reverse(ops);
    for (auto& op : ops) {
        adjoint_inplace(op);
    }
}

[[nodiscard]] inline CliffordOperatorString adjoint(CliffordOperatorString const& ops) {
    auto ret = ops;
    adjoint_inplace(ret);
    return ret;
}

/**
 * @brief Traits for Pauli Product-like classes. Such classes should implement the h, s, cx methods.
          The trait will generate the sdg, v, vdg, x, y, z, cz methods.
 * @tparam T should be the derived class.
 */
template <typename T>
class PauliProductTrait {
public:
    virtual ~PauliProductTrait() = default;

    virtual T& h(size_t qubit) noexcept                   = 0;
    virtual T& s(size_t qubit) noexcept                   = 0;
    virtual T& cx(size_t control, size_t target) noexcept = 0;

    T& sdg(size_t qubit) noexcept { return s(qubit).s(qubit).s(qubit); }
    T& v(size_t qubit) noexcept { return h(qubit).s(qubit).h(qubit); }
    T& vdg(size_t qubit) noexcept { return h(qubit).sdg(qubit).h(qubit); }

    T& x(size_t qubit) noexcept { return h(qubit).z(qubit).h(qubit); }
    T& y(size_t qubit) noexcept { return x(qubit).z(qubit); }
    T& z(size_t qubit) noexcept { return s(qubit).s(qubit); }
    T& cz(size_t control, size_t target) noexcept { return h(target).cx(control, target).h(target); }
    T& swap(size_t qubit1, size_t qubit2) noexcept { return cx(qubit1, qubit2).cx(qubit2, qubit1).cx(qubit1, qubit2); }
    T& ecr(size_t control, size_t target) noexcept { return cx(control, target).s(control).x(control).v(target); }

    T& apply(CliffordOperator const& op) {
        auto& [type, qubits] = op;
        switch (type) {
            case CliffordOperatorType::h:
                return h(qubits[0]);
            case CliffordOperatorType::s:
                return s(qubits[0]);
            case CliffordOperatorType::cx:
                return cx(qubits[0], qubits[1]);
            case CliffordOperatorType::sdg:
                return sdg(qubits[0]);
            case CliffordOperatorType::v:
                return v(qubits[0]);
            case CliffordOperatorType::vdg:
                return vdg(qubits[0]);
            case CliffordOperatorType::x:
                return x(qubits[0]);
            case CliffordOperatorType::y:
                return y(qubits[0]);
            case CliffordOperatorType::z:
                return z(qubits[0]);
            case CliffordOperatorType::cz:
                return cz(qubits[0], qubits[1]);
            case CliffordOperatorType::swap:
                return swap(qubits[0], qubits[1]);
            case CliffordOperatorType::ecr:
                return ecr(qubits[0], qubits[1]);
        }
        DVLAB_UNREACHABLE("Every Clifford type should be handled in the switch-case");
        return *static_cast<T*>(this);
    }

    T& apply(CliffordOperatorString const& ops) {
        for (auto const& op : ops) {
            this->apply(op);
        }
        return *static_cast<T*>(this);
    }
};

class PauliProduct : public PauliProductTrait<PauliProduct> {
public:
    PauliProduct(std::initializer_list<Pauli> const& pauli_list, bool is_neg);
    PauliProduct(std::string_view pauli_str);

    template <std::input_iterator I, std::sentinel_for<I> S>
    requires std::same_as<std::iter_value_t<I>, Pauli>
    PauliProduct(I first, S last, bool is_neg) : _bitset(2 * std::ranges::distance(first, last) + 1) {
        size_t i = 0;
        for (auto it = first; it != last; ++it) {
            set_pauli_type(i++, *it);
        }
        _bitset[_r_idx()] = is_neg;
    }

    template <std::ranges::range R>
    requires std::same_as<std::ranges::range_value_t<R>, Pauli>
    PauliProduct(R const& r, bool is_neg) : PauliProduct(std::ranges::begin(r), std::ranges::end(r), is_neg) {}

    size_t n_qubits() const { return (_bitset.size() - 1) / 2; }
    Pauli get_pauli_type(size_t i) const {
        return is_z_set(i) ? (is_x_set(i) ? Pauli::y : Pauli::z)
                           : (is_x_set(i) ? Pauli::x : Pauli::i);
    }

    bool is_i(size_t i) const { return !is_z_set(i) && !is_x_set(i); }
    bool is_x(size_t i) const { return !is_z_set(i) && is_x_set(i); }
    bool is_y(size_t i) const { return is_z_set(i) && is_x_set(i); }
    bool is_z(size_t i) const { return is_z_set(i) && !is_x_set(i); }

    /**
     * @brief Bitmask of qubits with a non-identity Pauli (X/Y/Z). Size is ``n_qubits()``.
     *
     * Built as the bitwise OR of the contiguous Z and X halves of the internal
     * encoding, so callers can use ``find_first`` / ``find_next`` / ``count``.
     */
    sul::dynamic_bitset<> non_identity_mask() const {
        auto const n = n_qubits();
        sul::dynamic_bitset<> mask(n);
        for (size_t i = 0; i < n; ++i) {
            if (_bitset[i] || _bitset[n + i]) {
                mask.set(i);
            }
        }
        return mask;
    }

    /// Number of non-identity Paulis (Pauli weight).
    size_t weight() const { return non_identity_mask().count(); }

    /// Ascending indices of qubits with a non-identity Pauli.
    std::vector<size_t> non_identity_qubits() const {
        auto const mask = non_identity_mask();
        std::vector<size_t> qubits;
        qubits.reserve(mask.count());
        for (auto i = mask.find_first(); i != sul::dynamic_bitset<>::npos; i = mask.find_next(i)) {
            qubits.push_back(i);
        }
        return qubits;
    }

    bool is_neg() const { return _bitset[_r_idx()]; }

    PauliProduct& operator*=(PauliProduct const& rhs);

    friend PauliProduct operator*(PauliProduct lhs, PauliProduct const& rhs) {
        lhs *= rhs;
        return lhs;
    }

    bool operator==(PauliProduct const& rhs) const { return _bitset == rhs._bitset; }
    bool operator!=(PauliProduct const& rhs) const { return _bitset != rhs._bitset; }

    std::string to_string(char signedness = '-') const;
    std::string to_bit_string() const;

    PauliProduct& h(size_t qubit) noexcept override;
    PauliProduct& s(size_t qubit) noexcept override;
    PauliProduct& cx(size_t control, size_t target) noexcept override;

    PauliProduct& negate() {
        _bitset.flip(_r_idx());
        return *this;
    }

    bool is_commutative(PauliProduct const& rhs) const;

    void set_pauli_type(size_t i, Pauli type) {
        switch (type) {
            case Pauli::i:
                _bitset[_z_idx(i)] = false;
                _bitset[_x_idx(i)] = false;
                break;
            case Pauli::x:
                _bitset[_z_idx(i)] = false;
                _bitset[_x_idx(i)] = true;
                break;
            case Pauli::y:
                _bitset[_z_idx(i)] = true;
                _bitset[_x_idx(i)] = true;
                break;
            case Pauli::z:
                _bitset[_z_idx(i)] = true;
                _bitset[_x_idx(i)] = false;
                break;
        }
    }

    bool is_z_set(size_t i) const { return _bitset[_z_idx(i)]; }
    bool is_x_set(size_t i) const { return _bitset[_x_idx(i)]; }

    void set_z(size_t i, bool z) { _bitset[_z_idx(i)] = z; }
    void set_x(size_t i, bool x) { _bitset[_x_idx(i)] = x; }

    bool is_diagonal() const {
        return std::ranges::all_of(std::views::iota(0ul, n_qubits()), [this](size_t i) { return is_i(i) || is_z(i); });
    }

    bool is_identity() const { return weight() == 0; }

    /** @brief Direct access to the underlying bitset (e.g. for hashing or low-level use). */
    sul::dynamic_bitset<> const& bitset() const { return _bitset; }

private:
    sul::dynamic_bitset<> _bitset;

    size_t _z_idx(size_t i) const { return i; }
    size_t _x_idx(size_t i) const { return i + n_qubits(); }
    size_t _r_idx() const { return n_qubits() * 2; }
};

inline bool is_commutative(PauliProduct const& lhs, PauliProduct const& rhs) {
    return lhs.is_commutative(rhs);
}

uint8_t power_of_i(PauliProduct const& lhs, PauliProduct const& rhs);

}  // namespace tableau
}  // namespace qsyn

namespace std {

template <>
struct hash<qsyn::tableau::PauliProduct> {
    size_t operator()(qsyn::tableau::PauliProduct const& p) const noexcept {
        auto const& b = p.bitset();
        size_t h      = std::hash<size_t>{}(b.size());
        for (size_t i = 0; i < b.size(); ++i) {
            h = (h * 31) + static_cast<size_t>(b[i]);
        }
        return h;
    }
};

}  // namespace std

template <>
struct fmt::formatter<qsyn::tableau::Pauli> {
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(qsyn::tableau::Pauli c, FormatContext& ctx) {
        string_view name = "I";
        switch (c) {
            case qsyn::tableau::Pauli::i:
                name = "I";
                break;
            case qsyn::tableau::Pauli::x:
                name = "X";
                break;
            case qsyn::tableau::Pauli::y:
                name = "Y";
                break;
            case qsyn::tableau::Pauli::z:
                name = "Z";
                break;
        }
        return fmt::format_to(ctx.out(), "{}", name);
    }
};

template <>
struct fmt::formatter<qsyn::tableau::PauliProduct> {
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
    auto format(qsyn::tableau::PauliProduct const& c, FormatContext& ctx) {
        if (presentation == 'b') {
            return fmt::format_to(ctx.out(), "{}", c.to_bit_string());
        } else {
            return fmt::format_to(ctx.out(), "{}", c.to_string(signedness));
        }
    }
};
