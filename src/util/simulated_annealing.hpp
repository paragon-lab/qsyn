/*
  PackageName  [ util ]
  Synopsis     [ Simulated annealing framework ]
  Author       [ Mu-Te (Joshua) Lau (joshmtlau), April Wang (april864) ]
*/

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>

namespace qsyn::util {

/**
 * @brief Simulated annealing framework.
 * @tparam T State type.
 * @tparam CostType Cost type.
 */
template <typename T, typename CostType>
struct SimulatedAnnealing {
    using MutateFn = std::function<void(T&)>;
    using CostFn   = std::function<CostType(T const&)>;
    using ProbType = double;

    /**
     * @brief Construct a new Simulated Annealing object with
     *        uniform probability for each mutate function.
     * @param init_temp Initial temperature.
     * @param cooling_rate Cooling rate.
     * @param min_temp Minimum temperature.
     * @param cost_fn Cost function.
     * @param mutate_fns Mutate functions.
     * @param seed Optional RNG seed. If omitted, uses ``std::random_device``.
     */
    SimulatedAnnealing(
        double init_temp,
        double cooling_rate,
        double min_temp,
        CostFn cost_fn,
        std::vector<MutateFn> mutate_fns,
        std::optional<std::uint32_t> seed = std::nullopt)
        : _init_temp(init_temp),
          _cooling_rate(cooling_rate),
          _min_temp(min_temp),
          _cost_fn(cost_fn),
          _mutate_fns(std::move(mutate_fns)),
          _gen(seed.has_value() ? std::mt19937{*seed} : std::mt19937{std::random_device{}()}) {
        auto const n_mutate_fns = _mutate_fns.size();

        std::vector<ProbType> mutate_fn_probs(n_mutate_fns, 1.0 / n_mutate_fns);
        _mutate_fn_dist = std::discrete_distribution<size_t>(mutate_fn_probs.begin(), mutate_fn_probs.end());
    }

    /**
     * @brief Construct a new Simulated Annealing object with custom probability for each mutate function.
     * @param init_temp Initial temperature.
     * @param cooling_rate Cooling rate.
     * @param min_temp Minimum temperature.
     * @param cost_fn Cost function.
     * @param mutate_fns Mutate functions.
     * @param mutate_fn_probs Probabilities for each mutate function.
     * @param seed Optional RNG seed. If omitted, uses ``std::random_device``.
     */
    SimulatedAnnealing(
        double init_temp,
        double cooling_rate,
        double min_temp,
        CostFn cost_fn,
        std::vector<MutateFn> mutate_fns,
        std::vector<ProbType> mutate_fn_probs,
        std::optional<std::uint32_t> seed = std::nullopt)
        : _init_temp(init_temp),
          _cooling_rate(cooling_rate),
          _min_temp(min_temp),
          _cost_fn(cost_fn),
          _mutate_fns(std::move(mutate_fns)),
          _gen(seed.has_value() ? std::mt19937{*seed} : std::mt19937{std::random_device{}()}) {
        _mutate_fn_dist = std::discrete_distribution<size_t>(mutate_fn_probs.begin(), mutate_fn_probs.end());
    }

    /**
     * @brief Run the simulated annealing algorithm.
     */
    std::pair<T, CostType>
    run(
        T const& initial_state,
        std::optional<size_t> num_iterations = std::nullopt) const {
        T current_state = initial_state;
        T best_state    = initial_state;

        CostType current_cost = _cost_fn(current_state);
        CostType best_cost    = current_cost;

        double temperature = _init_temp;

        auto iterations = 0;

        while (temperature > _min_temp) {
            auto test_state = current_state;
            _mutate(test_state);
            auto const new_cost = _cost_fn(test_state);
            if (_accept_state(new_cost, current_cost, temperature)) {
                current_state = std::move(test_state);
                current_cost  = new_cost;
                if (new_cost < best_cost) {
                    best_state = current_state;
                    best_cost  = new_cost;
                }
            }
            temperature *= _cooling_rate;
            iterations++;
            if (num_iterations && iterations >= *num_iterations) {
                break;
            }
        }

        return std::make_pair(best_state, best_cost);
    }

    std::pair<T, CostType>
    operator()(T const& initial_state, std::optional<size_t> num_iterations = std::nullopt) const {
        return run(initial_state, num_iterations);
    }

private:
    double _init_temp;
    double _cooling_rate;
    double _min_temp;

    CostFn _cost_fn;
    std::vector<MutateFn> _mutate_fns;
    std::discrete_distribution<size_t> mutable _mutate_fn_dist;
    std::mt19937 mutable _gen;

    std::uniform_real_distribution<ProbType> mutable _prob_dist{0.0, 1.0};

    bool _accept_state(CostType const& new_cost, CostType const& current_cost, double temperature) const {
        if (new_cost < current_cost) {
            return true;
        }
        return std::exp((current_cost - new_cost) / temperature) > _prob_dist(_gen);
    }

    void _mutate(T& state) const {
        _mutate_fns[_mutate_fn_dist(_gen)](state);
    }
};
}  // namespace qsyn::util
