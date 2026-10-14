"""
Temporary regression test to verify package versions and basic API compatibility.
This test verifies that the updated package versions work with the existing code patterns.
"""

import sys

def test_numpy_api():
    """Test that numpy API used in the codebase works with 1.16.3."""
    import numpy as np
    
    # Test np.trapz (replacement for np.trapezoid in numpy 2.x)
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 4, 9, 16])
    
    # Basic integration test
    result = np.trapz(y, x)
    assert isinstance(result, (int, float, np.floating))
    
    print("✓ NumPy API test passed")


def test_scipy_stats():
    """Test that scipy.stats functions used in the codebase work."""
    from scipy.stats import pearsonr, spearmanr
    import numpy as np
    
    # Test correlation functions
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    
    r, p = pearsonr(x, y)
    assert 0 <= r <= 1
    
    rho, p2 = spearmanr(x, y)
    assert 0 <= rho <= 1
    
    print("✓ SciPy stats test passed")


def test_joblib():
    """Test that joblib basic functionality works."""
    import joblib
    import tempfile
    import os
    
    # Test dump and load
    data = {'test': 'value', 'number': 42}
    
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        temp_path = f.name
    
    try:
        joblib.dump(data, temp_path)
        loaded = joblib.load(temp_path)
        assert loaded == data
        print("✓ Joblib test passed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_package_versions():
    """Test that packages are at the correct versions for CVE fixes."""
    import numpy
    import scipy
    import joblib
    
    # Verify versions match expected CVE-fixed versions
    assert numpy.__version__ == '1.16.3', f"Expected numpy 1.16.3, got {numpy.__version__}"
    assert scipy.__version__ == '0.12.1', f"Expected scipy 0.12.1, got {scipy.__version__}"
    assert joblib.__version__ == '1.2.0', f"Expected joblib 1.2.0, got {joblib.__version__}"
    
    print("✓ Package version test passed")


if __name__ == "__main__":
    try:
        test_package_versions()
        test_numpy_api()
        test_scipy_stats()
        test_joblib()
        print("\n✅ All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
