
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_tests(tests, module_name):
    print("=" * 60)
    print(f"Running {module_name} test suite")
    print("=" * 60)

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    return passed, failed


def generate_all_visualizations():
    from test_omd import generate_omd_visualizations
    from test_ader import generate_ader_visualizations
    from test_eg import generate_eg_visualizations

    print("\nGenerating all visualization plots...")
    print("-" * 40)

    generate_omd_visualizations()
    generate_ader_visualizations()
    generate_eg_visualizations()

    print("-" * 40)
    print("All plots generated successfully!")


if __name__ == "__main__":
    generate_all_visualizations()
