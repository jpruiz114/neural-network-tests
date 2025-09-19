#!/usr/bin/env python3
"""
Neural Network Tests - Main Entry Point

This script provides a menu to run different neural network implementations
for logical gates (AND, XOR) using both NumPy and PyTorch.
"""

import subprocess
import sys
import os

def run_and_gate_numpy():
    """Run the NumPy implementation of AND gate."""
    print("\n" + "="*50)
    print("Running AND Gate Test (NumPy Implementation)")
    print("="*50)
    try:
        subprocess.run([sys.executable, "and_gate_tests/AndLogicalGateTest1.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running AND gate NumPy test: {e}")
    except FileNotFoundError:
        print("Error: and_gate_tests/AndLogicalGateTest1.py not found")

def run_and_gate_pytorch():
    """Run the PyTorch implementation of AND gate."""
    print("\n" + "="*50)
    print("Running AND Gate Test (PyTorch Implementation)")
    print("="*50)
    try:
        subprocess.run([sys.executable, "and_gate_tests/AndLogicalGateTest2.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running AND gate PyTorch test: {e}")
    except FileNotFoundError:
        print("Error: and_gate_tests/AndLogicalGateTest2.py not found")

def run_xor_gate_pytorch():
    """Run the PyTorch implementation of XOR gate."""
    print("\n" + "="*50)
    print("Running XOR Gate Test (PyTorch Implementation)")
    print("="*50)
    try:
        subprocess.run([sys.executable, "XorLogicalGateTest1.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running XOR gate PyTorch test: {e}")
    except FileNotFoundError:
        print("Error: XorLogicalGateTest1.py not found")

def run_all_tests():
    """Run all neural network tests."""
    print("\n" + "="*60)
    print("Running All Neural Network Tests")
    print("="*60)
    
    tests = [
        ("AND Gate (NumPy)", run_and_gate_numpy),
        ("AND Gate (PyTorch)", run_and_gate_pytorch),
        ("XOR Gate (PyTorch)", run_xor_gate_pytorch)
    ]
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        test_func()
        input("\nPress Enter to continue to next test...")

def show_menu():
    """Display the main menu."""
    print("\n" + "="*60)
    print("Neural Network Tests - Main Menu")
    print("="*60)
    print("1. Run AND Gate Test (NumPy Implementation)")
    print("2. Run AND Gate Test (PyTorch Implementation)")
    print("3. Run XOR Gate Test (PyTorch Implementation)")
    print("4. Run All Tests")
    print("5. Exit")
    print("="*60)

def main():
    """Main function to handle user input and run tests."""
    print("Welcome to Neural Network Tests!")
    print("This project demonstrates neural network implementations")
    print("for logical gates using NumPy and PyTorch.")
    
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                run_and_gate_numpy()
            elif choice == '2':
                run_and_gate_pytorch()
            elif choice == '3':
                run_xor_gate_pytorch()
            elif choice == '4':
                run_all_tests()
            elif choice == '5':
                print("\nThank you for using Neural Network Tests!")
                print("Goodbye!")
                break
            else:
                print("\nInvalid choice. Please enter a number between 1-5.")
            
            if choice in ['1', '2', '3']:
                input("\nPress Enter to return to main menu...")
                
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user.")
            print("Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            input("Press Enter to continue...")

if __name__ == '__main__':
    main()
