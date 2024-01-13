import unittest
import os


def run_op_unittests():
    # Create the test suite
    suite = unittest.TestSuite()

    # Join the current working directory with the unittests folder
    unittests_dir = os.path.join(os.getcwd(), "micrograd", "unittests")
    print(f"Running unittests from {unittests_dir}")

    # Find all the test cases in this folder
    test_loader = unittest.TestLoader()
    test_loader.testMethodPrefix = "op_unittest_"
    tests = test_loader.discover(unittests_dir, pattern="op_unittests.py")

    # Print the tests
    print("Found the following tests:")
    # Print the tests
    for test_group in tests:
        for test in test_group:
            print(f"Test case: {test}")
            if isinstance(test, unittest.loader._FailedTest):
                print("Failed to load test methods")
                print(test)
                return
            else:
                for t in test:
                    print(f"Test method: {t}")

    # Add the tests to the suite
    suite.addTests(tests)

    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    print("Running all unittests...")
    run_op_unittests()
    print("Done.")
