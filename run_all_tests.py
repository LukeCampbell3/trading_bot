"""
Run all tests: Alpaca connection + Model accuracy
"""
import subprocess
import sys

def run_test(script_name, description):
    """Run a test script and return result"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    return result.returncode == 0

def main():
    print("\n" + "="*70)
    print("COMPLETE TEST SUITE")
    print("="*70)
    
    tests = [
        ("test_alpaca_connection.py", "Alpaca API Connection Tests"),
        ("test_model_accuracy.py", "Trading Model Accuracy Tests"),
    ]
    
    results = []
    for script, description in tests:
        try:
            passed = run_test(script, description)
            results.append((description, passed))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((description, False))
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    for description, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {description}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} test suites passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! System is ready for trading.")
        return 0
    else:
        print("\n⚠ Some tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
