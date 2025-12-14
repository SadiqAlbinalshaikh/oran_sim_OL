
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from learners import OMD, ADER
from test_base import PLOT_DIR, run_tests


def test_ader_initialization():
    T = 1000
    D = 1.0
    L = 1.0

    ader = ADER(dim=2, T=T, D=D, L=L)

    expected_N = 1 + int(np.ceil(np.log2(np.sqrt(4 + 8*T))))
    assert ader.N == expected_N, f"Wrong N: {ader.N} != {expected_N}"

    assert len(ader.experts) == expected_N, \
        f"Wrong number of experts: {len(ader.experts)} != {expected_N}"

    assert np.allclose(ader.weights, np.ones(ader.N) / ader.N), \
        "Initial weights should be uniform"


def test_ader_step_sizes():
    T = 1000
    D = 1.0
    L = 1.0

    ader = ADER(dim=2, T=T, D=D, L=L)

    expected_eta1 = D / (L * np.sqrt(T))
    assert np.isclose(ader.step_sizes[0], expected_eta1), \
        f"Wrong eta^(1): {ader.step_sizes[0]:.6f} != {expected_eta1:.6f}"

    for i in range(1, len(ader.step_sizes)):
        ratio = ader.step_sizes[i] / ader.step_sizes[i-1]
        assert np.isclose(ratio, 2.0), \
            f"Step sizes not geometric at index {i}: ratio={ratio:.6f}"


def test_ader_meta_learning_rate():
    T = 1000
    D = 2.0
    L = 1.5

    ader = ADER(dim=2, T=T, D=D, L=L)

    expected_beta = np.sqrt(2 * np.log(ader.N)) / (L * D * np.sqrt(T))
    assert np.isclose(ader.beta, expected_beta), \
        f"Wrong beta: {ader.beta:.6f} != {expected_beta:.6f}"


def test_ader_tracks_sinusoidal():
    np.random.seed(42)
    T = 1000
    D = 4.0
    L = 2.0
    amplitude = 1.5
    omega = 2 * np.pi / 200

    ader = ADER(dim=1, T=T, D=D, L=L)

    total_loss = 0.0
    for t in range(T):
        target = amplitude * np.sin(omega * t)
        x = ader.get_action()[0]
        loss = (x - target) ** 2
        total_loss += loss
        gradient = np.array([2 * (x - target)])
        ader.update(gradient)

    path_length = 0.0
    for t in range(1, T):
        prev_target = amplitude * np.sin(omega * (t - 1))
        curr_target = amplitude * np.sin(omega * t)
        path_length += abs(curr_target - prev_target)

    assert total_loss < T, \
        f"Regret too high for sinusoidal: {total_loss:.2f} >= {T}"


def test_ader_tracks_random_walk():
    np.random.seed(42)
    T = 1000
    D = 10.0
    L = 2.0
    sigma = 0.05

    ader = ADER(dim=1, T=T, D=D, L=L)

    target = 0.0
    total_loss = 0.0
    path_length = 0.0

    for t in range(T):
        x = ader.get_action()[0]
        loss = (x - target) ** 2
        total_loss += loss
        gradient = np.array([2 * (x - target)])
        ader.update(gradient)

        old_target = target
        target += sigma * np.random.randn()
        target = np.clip(target, -D/2, D/2)
        path_length += abs(target - old_target)

    expected_bound = 10 * np.sqrt(T * path_length)
    assert total_loss < expected_bound, \
        f"Regret too high for random walk: {total_loss:.2f} >= {expected_bound:.2f}"


def test_ader_weights_stable():
    np.random.seed(42)
    T = 500
    ader = ADER(dim=2, T=T, D=2.0, L=2.0)

    for t in range(T):
        gradient = np.array([np.sin(t * 0.1), np.cos(t * 0.1)])
        ader.update(gradient)

    assert np.all(ader.weights > 0), \
        f"Weights collapsed to zero: min={ader.weights.min()}"
    assert np.isclose(ader.weights.sum(), 1.0), \
        f"Weights don't sum to 1: sum={ader.weights.sum()}"


def test_ader_weights_adapt_to_dynamics():
    np.random.seed(42)
    T = 500
    D = 4.0
    L = 2.0

    ader_fast = ADER(dim=1, T=T, D=D, L=L)
    for t in range(T):
        target = 1.5 * np.sin(t * 0.2)
        x = ader_fast.get_action()[0]
        gradient = np.array([2 * (x - target)])
        ader_fast.update(gradient)

    ader_slow = ADER(dim=1, T=T, D=D, L=L)
    for t in range(T):
        target = 1.5 * np.sin(t * 0.01)
        x = ader_slow.get_action()[0]
        gradient = np.array([2 * (x - target)])
        ader_slow.update(gradient)

    initial_weights = np.ones(ader_fast.N) / ader_fast.N

    fast_entropy = -np.sum(ader_fast.weights * np.log(ader_fast.weights + 1e-10))
    slow_entropy = -np.sum(ader_slow.weights * np.log(ader_slow.weights + 1e-10))
    max_entropy = np.log(ader_fast.N)

    assert fast_entropy < max_entropy * 0.99, \
        f"Fast env weights didn't adapt: entropy={fast_entropy:.3f}, max={max_entropy:.3f}"
    assert slow_entropy < max_entropy * 0.99, \
        f"Slow env weights didn't adapt: entropy={slow_entropy:.3f}, max={max_entropy:.3f}"

    weight_diff = np.linalg.norm(ader_fast.weights - ader_slow.weights)
    assert weight_diff > 0.01, \
        f"Weights didn't adapt differently: diff={weight_diff:.4f}"


def test_ader_dynamic_regret_bound():
    np.random.seed(42)
    T = 1000
    D = 4.0
    L = 2.0

    def get_target(t):
        base = 0.5 * np.sin(t * 0.02)
        if t == 300 or t == 600:
            return base + 1.0
        return base

    ader = ADER(dim=1, T=T, D=D, L=L)

    total_loss = 0.0
    path_length = 0.0
    prev_target = get_target(0)

    for t in range(T):
        target = get_target(t)
        x = ader.get_action()[0]
        total_loss += (x - target) ** 2
        gradient = np.array([2 * (x - target)])
        ader.update(gradient)

        path_length += abs(target - prev_target)
        prev_target = target

    theoretical_bound = 3 * L * np.sqrt(T * (D**2 + 2 * D * path_length))

    assert total_loss < theoretical_bound * 2, \
        f"Dynamic regret exceeds bound: {total_loss:.1f} >= {2*theoretical_bound:.1f}"

    assert total_loss < 0.5 * T, \
        f"Regret not sublinear: {total_loss:.1f} >= {0.5*T}"


def test_ader_regret_scales_with_path_length():
    np.random.seed(42)
    T = 500
    D = 4.0
    L = 2.0

    path_lengths = []
    regrets = []

    for sigma in [0.01, 0.05, 0.1, 0.2]:
        np.random.seed(42)
        ader = ADER(dim=1, T=T, D=D, L=L)

        target = 0.0
        total_loss = 0.0
        path_length = 0.0

        for t in range(T):
            x = ader.get_action()[0]
            total_loss += (x - target) ** 2
            gradient = np.array([2 * (x - target)])
            ader.update(gradient)

            old_target = target
            target += sigma * np.random.randn()
            target = np.clip(target, -D/2, D/2)
            path_length += abs(target - old_target)

        path_lengths.append(path_length)
        regrets.append(total_loss)

    for i in range(1, len(regrets)):
        assert regrets[i] >= regrets[i-1] * 0.5, \
            f"Regret didn't increase with path length: {regrets}"


def test_ader_reset():
    ader = ADER(dim=2, T=100, D=1.0, L=1.0)
    initial_weights = ader.weights.copy()

    for t in range(20):
        gradient = np.array([np.sin(t * 0.1), np.cos(t * 0.1)])
        ader.update(gradient)

    ader.reset()

    assert np.allclose(ader.weights, initial_weights), \
        "Reset did not restore weights"


def plot_ader_sinusoidal():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    T = 1000
    D = 4.0
    L = 2.0
    amplitude = 1.5
    omega = 2 * np.pi / 200

    ader = ADER(dim=1, T=T, D=D, L=L)

    actions = []
    targets = []
    losses = []
    weights_history = []

    for t in range(T):
        target = amplitude * np.sin(omega * t)
        x = ader.get_action()[0]

        actions.append(x)
        targets.append(target)
        losses.append((x - target)**2)
        weights_history.append(ader.weights.copy())

        gradient = np.array([2 * (x - target)])
        ader.update(gradient)

    weights_history = np.array(weights_history)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax1 = axes[0]
    ax1.plot(targets, 'r-', linewidth=2, label='Sinusoidal target', alpha=0.8)
    ax1.plot(actions, 'b-', linewidth=1, label='ADER action', alpha=0.7)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'ADER Tracking Sinusoidal Target (period={int(2*np.pi/omega)} steps)', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(losses, 'g-', linewidth=0.5, alpha=0.5, label='Per-round loss')
    window = 50
    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax2.plot(np.arange(window-1, T), moving_avg, 'darkgreen', linewidth=2, label=f'Moving avg (window={window})')
    ax2.set_ylabel('Loss (x - target)^2', fontsize=12)
    ax2.set_title('Per-round Loss', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    colors = plt.cm.viridis(np.linspace(0, 1, ader.N))
    for i in range(ader.N):
        ax3.plot(weights_history[:, i], color=colors[i], linewidth=1,
                 label=f'eta={ader.step_sizes[i]:.4f}', alpha=0.8)
    ax3.set_xlabel('Round t', fontsize=12)
    ax3.set_ylabel('Expert Weight', fontsize=12)
    ax3.set_title('Expert Weights Adaptation', fontsize=14)
    ax3.legend(loc='upper right', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, 'ader_sinusoidal.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_ader_random_walk():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    T = 1000
    D = 20.0
    L = 50.0
    sigma = 0.5

    ader = ADER(dim=1, T=T, D=D, L=L)

    actions = []
    targets = []
    losses = []
    target = 0.0

    for t in range(T):
        x = ader.get_action()[0]

        actions.append(x)
        targets.append(target)
        losses.append((x - target)**2)

        gradient = np.array([2 * (x - target)])
        ader.update(gradient)

        target += sigma * np.random.randn()
        target = np.clip(target, -D/2, D/2)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax1 = axes[0]
    ax1.plot(targets, 'r-', linewidth=1.5, label='Random walk target', alpha=0.8)
    ax1.plot(actions, 'b-', linewidth=1, label='ADER action', alpha=0.7)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'ADER Tracking Random Walk (sigma={sigma})', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(losses, 'g-', linewidth=0.5, alpha=0.5, label='Per-round loss')
    window = 50
    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax2.plot(np.arange(window-1, T), moving_avg, 'darkgreen', linewidth=2, label=f'Moving avg (window={window})')
    ax2.set_ylabel('Loss (x - target)^2', fontsize=12)
    ax2.set_title('Per-round Loss', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    cumulative = np.cumsum(losses)
    ax3.plot(cumulative, 'purple', linewidth=2, label='ADER cumulative loss')
    t_vals = np.arange(1, T+1)
    ax3.plot(t_vals, 0.5 * t_vals, 'r--', linewidth=1.5, alpha=0.5, label='O(T) linear')
    ax3.plot(t_vals, 5 * np.sqrt(t_vals), 'g--', linewidth=1.5, alpha=0.5, label='O(sqrt(T)) sublinear')
    ax3.set_xlabel('Round t', fontsize=12)
    ax3.set_ylabel('Cumulative Loss', fontsize=12)
    ax3.set_title(f'Cumulative Loss: {cumulative[-1]:.1f}', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, 'ader_random_walk.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_ader_vs_omd():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    T = 1000
    D = 4.0
    L = 2.0

    def get_target(t):
        if t < T // 3:
            return 1.5 * np.sin(t * 0.01)
        elif t < 2 * T // 3:
            return 1.5 * np.sin(t * 0.15)
        else:
            return 1.5 * np.sin(t * 0.05)

    ader = ADER(dim=1, T=T, D=D, L=L)
    losses_ader = []
    actions_ader = []

    step_sizes = [0.05, 0.2, 0.5]
    omd_learners = [OMD(dim=1, step_size=eta) for eta in step_sizes]
    losses_omd = [[] for _ in step_sizes]

    targets = []
    for t in range(T):
        target = get_target(t)
        targets.append(target)

        x_ader = ader.get_action()[0]
        actions_ader.append(x_ader)
        losses_ader.append((x_ader - target)**2)
        ader.update(np.array([2 * (x_ader - target)]))

        for i, omd in enumerate(omd_learners):
            x = omd.get_action()[0]
            losses_omd[i].append((x - target)**2)
            omd.update(np.array([2 * (x - target)]))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax1 = axes[0]
    ax1.plot(targets, 'r-', linewidth=2, label='Target (varying frequency)', alpha=0.8)
    ax1.plot(actions_ader, 'b-', linewidth=1, label='ADER', alpha=0.7)
    ax1.axvline(x=T//3, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=2*T//3, color='gray', linestyle=':', alpha=0.5)
    ax1.text(T//6, 1.8, 'Slow', ha='center', fontsize=10, color='gray')
    ax1.text(T//2, 1.8, 'Fast', ha='center', fontsize=10, color='gray')
    ax1.text(5*T//6, 1.8, 'Medium', ha='center', fontsize=10, color='gray')
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Target with Changing Dynamics', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(np.cumsum(losses_ader), 'b-', linewidth=2, label='ADER (adaptive)')
    colors = ['orange', 'green', 'purple']
    for i, eta in enumerate(step_sizes):
        ax2.plot(np.cumsum(losses_omd[i]), color=colors[i], linestyle='--',
                 linewidth=1.5, label=f'OMD eta={eta}')
    ax2.axvline(x=T//3, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=2*T//3, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Cumulative Loss', fontsize=12)
    ax2.set_title('Cumulative Loss Comparison', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    phases = ['Slow\n(t<333)', 'Fast\n(333<=t<667)', 'Medium\n(t>=667)']
    phase_ranges = [(0, T//3), (T//3, 2*T//3), (2*T//3, T)]

    x_pos = np.arange(len(phases))
    width = 0.15

    ader_phase_losses = [sum(losses_ader[s:e]) for s, e in phase_ranges]
    omd_phase_losses = [[sum(losses_omd[i][s:e]) for s, e in phase_ranges] for i in range(len(step_sizes))]

    bars = ax3.bar(x_pos - 1.5*width, ader_phase_losses, width, label='ADER', color='blue')
    for i, eta in enumerate(step_sizes):
        ax3.bar(x_pos + (i-0.5)*width, omd_phase_losses[i], width, label=f'OMD eta={eta}', color=colors[i])

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(phases)
    ax3.set_ylabel('Phase Loss', fontsize=12)
    ax3.set_title('Per-Phase Performance (lower is better)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, 'ader_vs_omd.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_ader_path_length():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    T = 500
    D = 6.0
    L = 2.0

    sigmas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    path_lengths = []
    regrets = []

    for sigma in sigmas:
        np.random.seed(42)
        ader = ADER(dim=1, T=T, D=D, L=L)

        target = 0.0
        total_loss = 0.0
        path_length = 0.0

        for t in range(T):
            x = ader.get_action()[0]
            total_loss += (x - target) ** 2
            gradient = np.array([2 * (x - target)])
            ader.update(gradient)

            old_target = target
            target += sigma * np.random.randn()
            target = np.clip(target, -D/2, D/2)
            path_length += abs(target - old_target)

        path_lengths.append(path_length)
        regrets.append(total_loss)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(path_lengths, regrets, s=100, c='blue', zorder=5)
    for i, sigma in enumerate(sigmas):
        ax1.annotate(f'sigma={sigma}', (path_lengths[i], regrets[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)

    z = np.polyfit(np.sqrt(path_lengths), regrets, 1)
    p = np.poly1d(z)
    pl_sorted = np.sort(path_lengths)
    ax1.plot(pl_sorted, p(np.sqrt(pl_sorted)), 'r--', linewidth=2,
             label=f'Fitted: regret ~ {z[0]:.1f}*sqrt(P) + {z[1]:.1f}')

    ax1.set_xlabel('Path Length P', fontsize=12)
    ax1.set_ylabel('Dynamic Regret', fontsize=12)
    ax1.set_title('Regret vs Path Length', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    normalized = [r / np.sqrt(p) if p > 0 else 0 for r, p in zip(regrets, path_lengths)]
    ax2.bar(range(len(sigmas)), normalized, color='green', alpha=0.7)
    ax2.axhline(y=np.mean(normalized), color='red', linestyle='--',
                label=f'Mean: {np.mean(normalized):.1f}')
    ax2.set_xticks(range(len(sigmas)))
    ax2.set_xticklabels([f'sigma={s}' for s in sigmas])
    ax2.set_ylabel('Regret / sqrt(Path Length)', fontsize=12)
    ax2.set_title('Normalized Regret (should be ~constant)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('ADER Dynamic Regret Scales with sqrt(Path Length)', fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, 'ader_path_length.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_ader_visualizations():
    print("\nGenerating ADER plots...")
    plot_ader_sinusoidal()
    plot_ader_random_walk()
    plot_ader_vs_omd()
    plot_ader_path_length()


ADER_TESTS = [
    test_ader_initialization,
    test_ader_step_sizes,
    test_ader_meta_learning_rate,
    test_ader_tracks_sinusoidal,
    test_ader_tracks_random_walk,
    test_ader_weights_stable,
    test_ader_weights_adapt_to_dynamics,
    test_ader_dynamic_regret_bound,
    test_ader_regret_scales_with_path_length,
    test_ader_reset,
]


if __name__ == "__main__":
    passed, failed = run_tests(ADER_TESTS, "ADER")

    try:
        if '--plots' in sys.argv:
            generate_ader_visualizations()
    except:
        sys.exit(1)
    if failed > 0:
        sys.exit(1)

    print("\nAll ADER tests passed!")
    sys.exit(0)
