#!/usr/bin/env python3
"""Quick start script for DNATrainer tests.

This script provides a simple way to get started with testing
the DNATrainer class quickly and easily.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print a welcome banner."""
    print("🚀" * 20)
    print("  DNALLM DNATrainer Test Suite")
    print("  Quick Start Guide")
    print("🚀" * 20)

def check_environment():
    """Check the testing environment."""
    print("\n🔍 Checking environment...")
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the right directory
    if not (current_dir / "test_trainer.py").exists():
        print("❌ Please run this script from the tests/finetune directory")
        print("   cd tests/finetune")
        print("   python quick_start.py")
        return False
    
    print("✅ Environment check passed")
    return True

def check_dependencies():
    """Check available dependencies."""
    print("\n📦 Checking dependencies...")
    
    dependencies = {
        "torch": "PyTorch",
        "datasets": "Hugging Face Datasets",
        "transformers": "Hugging Face Transformers",
        "modelscope": "ModelScope",
        "pytest": "Pytest"
    }
    
    available = {}
    for dep, name in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {name} ({dep})")
            available[dep] = True
        except ImportError:
            print(f"❌ {name} ({dep}) - Not installed")
            available[dep] = False
    
    return available

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick test...")
    
    try:
        # Try to import and create a simple test
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Test basic imports
        from dnallm.finetune.trainer import DNATrainer
        print("✅ DNATrainer import successful")
        
        # Test configuration loading
        from dnallm.configuration.configs import load_config
        print("✅ Configuration module import successful")
        
        print("✅ Quick test passed - basic functionality works!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def show_test_options():
    """Show available testing options."""
    print("\n🎯 Available Testing Options:")
    print("1. 🚀 Quick Start (this script)")
    print("2. 🧪 Unit Tests Only")
    print("3. 🌐 Real Model Tests Only")
    print("4. 🔄 All Tests")
    print("5. 📖 Interactive Demo")
    print("6. 📚 Documentation")
    print("7. ❌ Exit")

def run_unit_tests():
    """Run unit tests."""
    print("\n🧪 Running unit tests...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "test_trainer.py", "-v"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Unit tests completed successfully!")
        print("Output preview:")
        print(result.stdout[-500:] + "..." if len(result.stdout) > 500 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Unit tests failed!")
        if e.stderr:
            print("Error:", e.stderr)
        return False

def run_real_model_tests():
    """Run real model tests."""
    print("\n🌐 Running real model tests...")
    
    try:
        result = subprocess.run(
            [sys.executable, "test_trainer_real_model.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Real model tests completed successfully!")
        print("Output preview:")
        print(result.stdout[-500:] + "..." if len(result.stdout) > 500 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Real model tests failed!")
        if e.stderr:
            print("Error:", e.stderr)
        return False

def run_all_tests():
    """Run all tests."""
    print("\n🔄 Running all tests...")
    
    success1 = run_unit_tests()
    success2 = run_real_model_tests()
    
    if success1 and success2:
        print("🎉 All tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return success1 and success2

def show_documentation():
    """Show documentation and help."""
    print("\n📚 Documentation and Help:")
    print("=" * 50)
    
    print("\n📁 Test Files:")
    print("  • test_trainer.py - Unit tests with mocked objects")
    print("  • test_trainer_real_model.py - Integration tests with real models")
    print("  • test_finetune_config.yaml - Test configuration")
    print("  • test_runner.py - Automated test runner")
    print("  • example_usage.py - Interactive usage examples")
    print("  • README.md - Detailed documentation")
    
    print("\n🚀 Quick Commands:")
    print("  • python -m unittest test_trainer.py -v")
    print("  • python test_trainer_real_model.py")
    print("  • python test_runner.py")
    print("  • python example_usage.py")
    
    print("\n💡 Tips:")
    print("  • Unit tests run quickly and don't need internet")
    print("  • Real model tests require ModelScope and internet")
    print("  • Check README.md for detailed instructions")
    print("  • Use test_runner.py for automated testing")
    
    print("\n🔧 Dependencies:")
    print("  • torch - PyTorch for deep learning")
    print("  • datasets - Hugging Face datasets")
    print("  • transformers - Hugging Face transformers")
    print("  • modelscope - ModelScope for model access")
    print("  • pytest - Testing framework (optional)")

def interactive_menu():
    """Interactive menu for test selection."""
    while True:
        show_test_options()
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                print("\n🚀 Quick Start - Running basic checks...")
                check_environment()
                deps = check_dependencies()
                run_quick_test()
                
            elif choice == "2":
                run_unit_tests()
                
            elif choice == "3":
                deps = check_dependencies()
                if deps.get("modelscope", False):
                    run_real_model_tests()
                else:
                    print("❌ ModelScope not available. Install with: pip install modelscope")
                    
            elif choice == "4":
                run_all_tests()
                
            elif choice == "5":
                print("\n📖 Starting interactive demo...")
                if os.path.exists("example_usage.py"):
                    try:
                        subprocess.run([sys.executable, "example_usage.py"])
                    except Exception as e:
                        print(f"❌ Demo failed: {e}")
                else:
                    print("❌ Demo script not found")
                    
            elif choice == "6":
                show_documentation()
                
            elif choice == "7":
                print("\n👋 Goodbye! Happy testing!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main function."""
    print_banner()
    
    if not check_environment():
        return
    
    deps = check_dependencies()
    
    print("\n🎯 Quick Start Options:")
    print("1. 🚀 Run all tests automatically")
    print("2. 🎮 Interactive menu")
    print("3. 📖 Show documentation")
    print("4. ❌ Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Running all tests automatically...")
            run_all_tests()
            
        elif choice == "2":
            interactive_menu()
            
        elif choice == "3":
            show_documentation()
            
        elif choice == "4":
            print("\n👋 Goodbye! Happy testing!")
            
        else:
            print("❌ Invalid choice. Starting interactive menu...")
            interactive_menu()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleanup completed")
        print("👋 Exiting quick start script...")
