#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
import time

class TestConfig:
    """Configuration for a layer test"""
    def __init__(self, name, layer_dir, test_script, make_target="sw_emu"):
        self.name = name
        self.layer_dir = layer_dir
        self.test_script = test_script
        self.make_target = make_target

class TestResult:
    """Results from a test"""
    def __init__(self, name, passed, error=None):
        self.name = name
        self.passed = passed
        self.error = error
    
    def __str__(self):
        if self.passed:
            return f"✅ {self.name}: PASSED"
        else:
            return f"❌ {self.name}: FAILED - {self.error}"

class TestRunner:
    """Main test runner class"""
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.tests = []
        self.results = []
    
    def add_test(self, test_config):
        """Add a test to run"""
        self.tests.append(test_config)
    
    def run_all_tests(self):
        """Run all registered tests"""
        for test in self.tests:
            print(f"\n{'-' * 80}")
            print(f"Running test: {test.name}")
            print(f"{'-' * 80}")
            
            try:
                result = self.run_test(test)
                self.results.append(result)
                print(result)
            except Exception as e:
                result = TestResult(test.name, False, f"Exception: {str(e)}")
                self.results.append(result)
                print(result)
            
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        for result in self.results:
            print(result)
        
        print("-" * 80)
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        print("=" * 80)
    
    def run_test(self, test_config):
        """Run a specific test"""
        # Change to the correct directory
        original_dir = os.getcwd()
        os.chdir(self.project_root / test_config.layer_dir)
        
        try:
            # Step 1: Run the test script to generate input data
            test_script = test_config.test_script
            print(f"Running test script to generate input data: {test_script}")
            
            # Start the Python test script
            process = subprocess.Popen(
                ["python", test_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Wait for the script to generate input data and prompt for user input
            output = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                    
                print(line, end='')
                output.append(line)
                
                # Check if the script is waiting for user input to continue
                if "Press Enter" in line or "press Enter" in line:
                    break
            
            # Now pause the test script and let the user run make
            print("\n" + "=" * 80)
            print(f"Input files have been generated. Now you can run make in this terminal:")
            print(f"$ make run TARGET={test_config.make_target}")
            print("=" * 80 + "\n")
            
            # Give control back to the user to run make commands
            subprocess_cmd = input("Press Enter to run the make command automatically, or type 'manual' to run commands yourself: ")
            
            if subprocess_cmd.strip().lower() != 'manual':
                # Run make command for the user
                print(f"\nRunning: make run TARGET={test_config.make_target}")
                print("-" * 80)
                make_result = subprocess.run(
                    ["make", "run", f"TARGET={test_config.make_target}"],
                    text=True
                )
                print("-" * 80)
                
                if make_result.returncode != 0:
                    return TestResult(test_config.name, False, f"Make command failed with exit code {make_result.returncode}")
                
                print("Make command completed. Continuing with test...")
            else:
                print("\nYou selected manual mode. Please run your commands now.")
                print("When finished, press Enter to continue with the test.")
                input("\nPress Enter to continue the test after running your commands...")
            
            # Send Enter to the Python process to continue
            process.stdin.write("\n")
            process.stdin.flush()
            
            # Continue reading output until the process completes
            remaining_output = process.communicate()[0]
            print(remaining_output)
            output.append(remaining_output)
            
            # Check if the test passed
            all_output = ''.join(output)
            if "Test FAILED" in all_output or "Mismatched elements" in all_output:
                return TestResult(test_config.name, False, "Test comparison failed. See output above for details.")
            elif "Test Passed" in all_output or "test passed" in all_output.lower():
                return TestResult(test_config.name, True)
            else:
                return TestResult(test_config.name, False, "Could not determine if test passed or failed.")
                
        finally:
            # Return to original directory
            os.chdir(original_dir)


def create_config_file(config_path):
    """Create a default configuration file if it doesn't exist"""
    if os.path.exists(config_path):
        return
    
    default_config = {
        "project_root": ".",
        "tests": [
            {
                "name": "convtranspose",
                "layer_dir": "layers/convtranspose",
                "test_script": "test_convtranspose.py",
                "make_target": "sw_emu"
            },
            {
                "name": "relu",
                "layer_dir": "layers/relu",
                "test_script": "test_relu.py",
                "make_target": "sw_emu"
            },
            {
                "name": "tanh",
                "layer_dir": "layers/tanh",
                "test_script": "test_tanh.py",
                "make_target": "sw_emu"
            },
            {
                "name": "batchnorm",
                "layer_dir": "layers/batchnorm",
                "test_script": "test_batchnorm.py",
                "make_target": "sw_emu"
            }
        ]
    }
    
    with open(config_path, "w") as f:
        json.dump(default_config, f, indent=4)
    
    print(f"Created default configuration file at {config_path}")
    print("Please edit this file to match your project structure and test scripts.")

def main():
    parser = argparse.ArgumentParser(description="FPGA Layer Testing Framework")
    parser.add_argument("--config", default="fpga_test_config.json", help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true", help="Create a default configuration file")
    parser.add_argument("--test", help="Run a specific test by name")
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        create_config_file(args.config)
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Run with --create-config to create a default configuration file.")
        return
    
    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Create test runner
    runner = TestRunner(config["project_root"])
    
    # Add tests
    for test_config in config["tests"]:
        # Skip tests not specified if --test is used
        if args.test and args.test != test_config["name"]:
            continue
            
        runner.add_test(TestConfig(
            test_config["name"],
            test_config["layer_dir"],
            test_config["test_script"],
            test_config.get("make_target", "sw_emu")
        ))
    
    # Run tests
    if runner.tests:
        runner.run_all_tests()
    else:
        if args.test:
            print(f"Test '{args.test}' not found in configuration.")
        else:
            print("No tests configured.")

if __name__ == "__main__":
    main()