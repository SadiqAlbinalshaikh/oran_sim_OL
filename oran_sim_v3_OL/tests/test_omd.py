
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from learners import OMD
from test_base import PLOT_DIR, run_tests


def test_omd_convergence_quadratic():
    np.random.seed(42)
    x_star = np.array([1.0, 2.0])
    omd = OMD(dim=2, step_size=0.1)

    for t in range(100):
        x = omd.get_action()
        gradient = 2 * (x - x_star)
        omd.update(gradient)

    error = np.linalg.norm(omd.get_action() - x_star)
    assert error < 0.1, f"OMD did not converge: error={error:.6f}"


def test_omd_projection():
    def project_to_ball(x, radius=1.0):
        norm = np.linalg.norm(x)
        return x if norm <= radius else x * radius / norm

    omd = OMD(dim=2, step_size=1.0, constraint_set=project_to_ball)

    omd.update(np.array([10.0, 10.0]))

    norm = np.linalg.norm(omd.get_action())
    assert norm <= 1.0 + 1e-6, f"Projection violated: norm={norm:.6f}"


def test_omd_reset():
    x0 = np.array([0.5, 0.5])
    omd = OMD(dim=2, step_size=0.1, x0=x0)

    for _ in range(10):
        omd.update(np.random.randn(2))

    omd.reset()
    assert np.allclose(omd.get_action(), x0), "Reset did not restore initial state"
    assert np.allclose(omd.prev_hint, np.zeros(2)), "Reset did not clear hint"


def test_optimistic_omd_with_hints():
    np.random.seed(42)
    T = 200

    def get_target(t):
        return np.array([np.sin(t * 0.02), np.cos(t * 0.02)])

    omd_std = OMD(dim=2, step_size=0.1)
    loss_std = 0.0
    for t in range(T):
        x = omd_std.get_action()
        target = get_target(t)
        gradient = 2 * (x - target)
        loss_std += np.sum((x - target)**2)
        omd_std.update(gradient)

    omd_opt = OMD(dim=2, step_size=0.1)
    loss_opt = 0.0
    prev_grad = np.zeros(2)
    for t in range(T):
        x = omd_opt.get_action()
        target = get_target(t)
        gradient = 2 * (x - target)
        loss_opt += np.sum((x - target)**2)
        omd_opt.update(gradient, hint=prev_grad)
        prev_grad = gradient.copy()

    assert loss_opt <= loss_std * 1.1, \
        f"Optimistic worse than standard: {loss_opt:.2f} > {loss_std:.2f}"


def test_omd_tracking_moving_target():
    np.random.seed(42)
    T = 500
    velocity = 0.01

    omd = OMD(dim=1, step_size=0.5)

    errors = []
    for t in range(T):
        target = velocity * t
        x = omd.get_action()[0]
        errors.append(abs(x - target))
        gradient = np.array([2 * (x - target)])
        omd.update(gradient)

    steady_state_errors = errors[100:]
    avg_error = np.mean(steady_state_errors)
    assert avg_error < 1.0, f"Tracking error too large: {avg_error:.4f}"


def plot_omd_convergence():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    x_star = np.array([1.0, 2.0])
    omd = OMD(dim=2, step_size=0.1)

    trajectory = [omd.get_action().copy()]
    for t in range(100):
        x = omd.get_action()
        gradient = 2 * (x - x_star)
        omd.update(gradient)
        trajectory.append(omd.get_action().copy())

    trajectory = np.array(trajectory)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='OMD trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, zorder=5, label='Start (0, 0)')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='blue', s=100, zorder=5, label=f'Final ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})')
    ax.scatter(x_star[0], x_star[1], c='red', s=150, marker='*', zorder=5, label=f'Optimal x* ({x_star[0]}, {x_star[1]})')

    x_range = np.linspace(-0.5, 1.5, 100)
    y_range = np.linspace(-0.5, 2.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X - x_star[0])**2 + (Y - x_star[1])**2
    ax.contour(X, Y, Z, levels=10, colors='gray', alpha=0.3)

    ax.set_xlabel('x_1', fontsize=12)
    ax.set_ylabel('x_2', fontsize=12)
    ax.set_title('OMD Convergence: minimize ||x - x*||^2', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, 'omd_convergence.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_omd_tracking_moving():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    T = 300

    def get_target(t):
        return np.array([np.sin(t * 0.03), np.cos(t * 0.03)])

    omd = OMD(dim=2, step_size=0.3)

    actions = []
    targets = []
    errors = []

    for t in range(T):
        target = get_target(t)
        x = omd.get_action()
        actions.append(x.copy())
        targets.append(target.copy())
        errors.append(np.linalg.norm(x - target))
        gradient = 2 * (x - target)
        omd.update(gradient)

    actions = np.array(actions)
    targets = np.array(targets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(targets[:, 0], targets[:, 1], 'r-', linewidth=2, label='Moving target', alpha=0.8)
    ax1.plot(actions[:, 0], actions[:, 1], 'b--', linewidth=1.5, label='OMD tracking', alpha=0.7)
    ax1.scatter(targets[0, 0], targets[0, 1], c='red', s=100, marker='o', zorder=5, label='Target start')
    ax1.scatter(actions[0, 0], actions[0, 1], c='blue', s=100, marker='s', zorder=5, label='OMD start')
    ax1.set_xlabel('x_1', fontsize=12)
    ax1.set_ylabel('x_2', fontsize=12)
    ax1.set_title('OMD Tracking Circular Target', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    ax2 = axes[1]
    ax2.plot(errors, 'g-', linewidth=1.5)
    ax2.fill_between(range(T), errors, alpha=0.3, color='green')
    ax2.axhline(y=np.mean(errors[50:]), color='k', linestyle='--', label=f'Avg error (t>50): {np.mean(errors[50:]):.3f}')
    ax2.set_xlabel('Round t', fontsize=12)
    ax2.set_ylabel('Tracking Error ||x - target||', fontsize=12)
    ax2.set_title('Tracking Error Over Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, 'omd_tracking_moving.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_omd_visualizations():
    print("\nGenerating OMD plots...")
    plot_omd_convergence()
    plot_omd_tracking_moving()


OMD_TESTS = [
    test_omd_convergence_quadratic,
    test_omd_projection,
    test_omd_reset,
    test_optimistic_omd_with_hints,
    test_omd_tracking_moving_target,
]


if __name__ == "__main__":
    passed, failed = run_tests(OMD_TESTS, "OMD")

    try:
        if '--plots' in sys.argv:
            generate_omd_visualizations()
    except:
        sys.exit(1)
    if failed > 0:
        sys.exit(1)

    print("\nAll OMD tests passed!")
    sys.exit(0)
