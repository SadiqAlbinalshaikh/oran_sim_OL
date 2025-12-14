
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from learners import EG
from test_base import PLOT_DIR, run_tests


def test_eg_initialization():
    n_arms = 5
    eg = EG(n_arms=n_arms, step_size=0.1)

    assert eg.n_arms == n_arms, f"Expected {n_arms} arms, got {eg.n_arms}"
    assert eg.dim == n_arms, f"dim should equal n_arms"
    assert len(eg.weights) == n_arms, f"weights length mismatch"
    assert np.allclose(eg.weights, np.ones(n_arms)), "Initial weights should be uniform"


def test_eg_normalization():
    eg = EG(n_arms=4, step_size=0.1)

    dist = eg.get_distribution()
    assert np.isclose(dist.sum(), 1.0), f"Distribution doesn't sum to 1: {dist.sum()}"
    assert np.allclose(dist, 0.25 * np.ones(4)), "Initial should be uniform 0.25"

    for _ in range(10):
        losses = np.random.rand(4)
        eg.update(losses)
        dist = eg.get_distribution()
        assert np.isclose(dist.sum(), 1.0), f"Distribution doesn't sum to 1: {dist.sum()}"
        assert np.all(dist >= 0), "Distribution has negative values"


def test_eg_multiplicative_update():
    eta = 0.5
    eg = EG(n_arms=3, step_size=eta, min_weight=1e-20)

    initial_weights = eg.weights.copy()

    losses = np.array([1.0, 0.0, 0.5])
    eg.update(losses)

    expected = initial_weights * np.exp(-eta * losses)

    assert np.allclose(eg.weights, expected), \
        f"Multiplicative update incorrect: got {eg.weights}, expected {expected}"


def test_eg_converges_to_best_arm():
    np.random.seed(42)
    n_arms = 5
    best_arm = 2
    T = 500

    eta = np.sqrt(2 * np.log(n_arms) / T)
    eg = EG(n_arms=n_arms, step_size=eta)

    for t in range(T):
        losses = np.ones(n_arms)
        losses[best_arm] = 0.0
        eg.update(losses)

    dist = eg.get_distribution()
    assert dist[best_arm] > 0.9, \
        f"Best arm should have >90% mass, got {dist[best_arm]:.4f}"


def test_eg_weights_stable():
    min_weight = 0.01
    eg = EG(n_arms=3, step_size=1.0, min_weight=min_weight)

    for _ in range(100):
        losses = np.array([10.0, 0.0, 10.0])
        eg.update(losses)

    assert np.all(eg.weights >= min_weight), \
        f"Weights collapsed below min_weight: {eg.weights}"


def test_eg_add_remove_arm():
    eg = EG(n_arms=3, step_size=0.1)

    eg.add_arm(initial_weight=1.0)
    assert eg.n_arms == 4, f"Expected 4 arms after add, got {eg.n_arms}"
    assert len(eg.weights) == 4, f"Weight vector length mismatch"

    eg.remove_arm(index=1)
    assert eg.n_arms == 3, f"Expected 3 arms after remove, got {eg.n_arms}"
    assert len(eg.weights) == 3, f"Weight vector length mismatch"

    dist = eg.get_distribution()
    assert np.isclose(dist.sum(), 1.0), f"Distribution invalid after arm changes"


def test_eg_reset():
    eg = EG(n_arms=4, step_size=0.5)

    for _ in range(50):
        eg.update(np.random.rand(4))

    assert not np.allclose(eg.weights, np.ones(4)), "Weights didn't change"

    eg.reset()
    assert np.allclose(eg.weights, np.ones(4)), "Reset didn't restore uniform"


def test_eg_regret_bound():
    np.random.seed(42)
    n_arms = 10
    T = 1000

    eta = np.sqrt(2 * np.log(n_arms) / T)
    eg = EG(n_arms=n_arms, step_size=eta)

    all_losses = np.random.rand(T, n_arms)

    cumulative_loss = 0.0
    for t in range(T):
        dist = eg.get_distribution()
        losses = all_losses[t]

        cumulative_loss += np.dot(dist, losses)
        eg.update(losses)

    arm_total_losses = all_losses.sum(axis=0)
    best_arm_loss = arm_total_losses.min()

    regret = cumulative_loss - best_arm_loss
    theoretical_bound = np.sqrt(2 * T * np.log(n_arms))

    assert regret < 2 * theoretical_bound, \
        f"Regret {regret:.2f} exceeds 2x bound {2*theoretical_bound:.2f}"


def test_eg_equal_losses_stay_uniform():
    eg = EG(n_arms=4, step_size=0.1)

    for _ in range(100):
        losses = 0.5 * np.ones(4)
        eg.update(losses)

    dist = eg.get_distribution()
    assert np.allclose(dist, 0.25 * np.ones(4), atol=1e-6), \
        f"Distribution should stay uniform: {dist}"


def plot_eg_convergence():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n_arms = 5
    best_arm = 2
    T = 200

    eta = np.sqrt(2 * np.log(n_arms) / T)
    eg = EG(n_arms=n_arms, step_size=eta)

    weight_history = [eg.get_distribution().copy()]

    for t in range(T):
        losses = np.ones(n_arms) * 0.8
        losses[best_arm] = 0.2
        losses += 0.1 * np.random.randn(n_arms)
        losses = np.clip(losses, 0, 1)
        eg.update(losses)
        weight_history.append(eg.get_distribution().copy())

    weight_history = np.array(weight_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, n_arms))
    for i in range(n_arms):
        label = f'Arm {i}' + (' (best)' if i == best_arm else '')
        linewidth = 2.5 if i == best_arm else 1.5
        ax.plot(weight_history[:, i], label=label, color=colors[i], linewidth=linewidth)

    ax.set_xlabel('Round t', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('EG Weight Evolution: Convergence to Best Arm', fontsize=14)
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, 'eg_convergence.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_eg_adversarial():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n_arms = 3
    T = 300

    eta = np.sqrt(2 * np.log(n_arms) / T)
    eg = EG(n_arms=n_arms, step_size=eta)

    weight_history = [eg.get_distribution().copy()]
    loss_history = []
    best_arm_history = []

    for t in range(T):
        best_arm = (t // 100) % n_arms

        losses = np.ones(n_arms) * 0.8
        losses[best_arm] = 0.2
        loss_history.append(losses.copy())
        best_arm_history.append(best_arm)

        eg.update(losses)
        weight_history.append(eg.get_distribution().copy())

    weight_history = np.array(weight_history)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(n_arms):
        ax1.plot(weight_history[:, i], label=f'Arm {i}', color=colors[i], linewidth=1.5)

    for change_t in [100, 200]:
        ax1.axvline(x=change_t, color='gray', linestyle='--', alpha=0.5)

    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('EG on Adversarial Sequence: Best Arm Changes Every 100 Rounds', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2 = axes[1]
    losses_array = np.array(loss_history)
    weight_array = weight_history[:-1]

    eg_cumulative = np.cumsum(np.sum(weight_array * losses_array, axis=1))
    best_arm_cumulative = np.cumsum([losses_array[t, best_arm_history[t]] for t in range(T)])
    uniform_cumulative = np.cumsum(np.mean(losses_array, axis=1))

    ax2.plot(eg_cumulative, label='EG', linewidth=2, color='blue')
    ax2.plot(best_arm_cumulative, label='Best-in-hindsight', linewidth=2, color='red', linestyle='--')
    ax2.plot(uniform_cumulative, label='Uniform', linewidth=1.5, color='gray', linestyle=':')

    ax2.set_xlabel('Round t', fontsize=12)
    ax2.set_ylabel('Cumulative Loss', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, 'eg_adversarial.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_eg_regret():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n_arms = 10
    T = 2000

    eta = np.sqrt(2 * np.log(n_arms) / T)
    eg = EG(n_arms=n_arms, step_size=eta)

    all_losses = np.random.rand(T, n_arms)
    eg_losses = []

    for t in range(T):
        dist = eg.get_distribution()
        losses = all_losses[t]
        eg_losses.append(np.dot(dist, losses))
        eg.update(losses)

    eg_cumulative = np.cumsum(eg_losses)

    arm_total_losses = all_losses.cumsum(axis=0)
    best_arm_cumulative = np.array([arm_total_losses[t].min() for t in range(T)])

    regret = eg_cumulative - best_arm_cumulative

    t_range = np.arange(1, T + 1)
    theoretical_bound = np.sqrt(2 * t_range * np.log(n_arms))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(regret, label='EG Regret', linewidth=2, color='blue')
    ax1.plot(theoretical_bound, label=r'$\sqrt{2T\ln(n)}$ bound', linewidth=2,
             color='red', linestyle='--')
    ax1.fill_between(range(T), 0, theoretical_bound, alpha=0.1, color='red')

    ax1.set_xlabel('Round T', fontsize=12)
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.set_title('EG Regret vs Theoretical Bound', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    normalized_regret = regret / np.sqrt(t_range)
    theoretical_normalized = np.sqrt(2 * np.log(n_arms)) * np.ones(T)

    ax2.plot(normalized_regret, label='Regret / sqrt(T)', linewidth=2, color='blue')
    ax2.axhline(y=theoretical_normalized[0], color='red', linestyle='--',
                label=r'$\sqrt{2\ln(n)}$ = ' + f'{theoretical_normalized[0]:.2f}')

    ax2.set_xlabel('Round T', fontsize=12)
    ax2.set_ylabel('Regret / sqrt(T)', fontsize=12)
    ax2.set_title('Normalized Regret (should stabilize)', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, 'eg_regret.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_eg_visualizations():
    print("\nGenerating EG plots...")
    plot_eg_convergence()
    plot_eg_adversarial()
    plot_eg_regret()


EG_TESTS = [
    test_eg_initialization,
    test_eg_normalization,
    test_eg_multiplicative_update,
    test_eg_converges_to_best_arm,
    test_eg_weights_stable,
    test_eg_add_remove_arm,
    test_eg_reset,
    test_eg_regret_bound,
    test_eg_equal_losses_stay_uniform,
]


if __name__ == "__main__":
    passed, failed = run_tests(EG_TESTS, "EG")

    try:
        if '--plots' in sys.argv:
            generate_eg_visualizations()
    except Exception as e:
        print(f"Error generating plots: {e}")
        sys.exit(1)

    if failed > 0:
        sys.exit(1)

    print("\nAll EG tests passed!")
    sys.exit(0)
