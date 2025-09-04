#!/bin/bash

# Song Editor 3 - PyInstaller Build Script
# Creates a standalone executable for deployment

set -e  # Exit on any error

echo "🎵 Building Song Editor 3 Standalone Executable"
echo "==============================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found. Setting up..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    fi

    # Check if PyInstaller is installed
    if ! python -c "import PyInstaller" 2>/dev/null; then
        print_warning "PyInstaller not found. Installing..."
        pip install pyinstaller
    fi

    # Check for required system dependencies
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS specific checks
        if ! command -v brew &> /dev/null; then
            print_warning "Homebrew not found. Some features may not work optimally."
        fi
    fi

    print_status "Prerequisites check complete"
}

# Clean previous builds
clean_build() {
    print_info "Cleaning previous builds..."
    rm -rf build dist *.spec
    print_status "Clean complete"
}

# Create or update spec file
prepare_spec() {
    print_info "Preparing PyInstaller spec file..."

    if [ ! -f "song_editor_3.spec" ]; then
        print_info "Creating spec file..."
        python build_app.py
    else
        print_info "Using existing spec file"
    fi

    print_status "Spec file ready"
}

# Build the executable
build_executable() {
    print_info "Building standalone executable..."
    print_info "This may take several minutes depending on system performance..."

    # Set environment variables for better build
    export PYTHONDONTWRITEBYTECODE=1

    # Build with PyInstaller
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_info "Building for macOS..."
        pyinstaller --clean --noconfirm song_editor_3.spec

    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        print_info "Building for Windows..."
        pyinstaller --clean --noconfirm --noconsole song_editor_3.spec

    else
        print_info "Building for Linux..."
        pyinstaller --clean --noconfirm song_editor_3.spec
    fi

    if [ $? -eq 0 ]; then
        print_status "Build completed successfully!"
    else
        print_error "Build failed!"
        exit 1
    fi
}

# Verify the build
verify_build() {
    print_info "Verifying build..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -d "dist/Song Editor 3.app" ]; then
            print_status "macOS app bundle created successfully"
            ls -la "dist/Song Editor 3.app/Contents/MacOS/"
        else
            print_error "macOS app bundle not found"
            return 1
        fi

    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        if [ -f "dist/SongEditor3.exe" ]; then
            print_status "Windows executable created successfully"
            ls -la dist/SongEditor3.exe
        else
            print_error "Windows executable not found"
            return 1
        fi

    else
        if [ -f "dist/SongEditor3" ]; then
            print_status "Linux executable created successfully"
            ls -la dist/SongEditor3
        else
            print_error "Linux executable not found"
            return 1
        fi
    fi

    # Check bundle size
    print_info "Build size:"
    du -sh dist/

    return 0
}

# Create distribution package
create_distribution() {
    print_info "Creating distribution package..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v create-dmg &> /dev/null; then
            print_info "Creating DMG for macOS distribution..."
            create-dmg \
                --volname "Song Editor 3" \
                --volicon "icon.icns" \
                --window-pos 200 120 \
                --window-size 800 400 \
                --icon-size 100 \
                --icon "Song Editor 3.app" 200 190 \
                --hide-extension "Song Editor 3.app" \
                --app-drop-link 600 185 \
                --no-internet-enable \
                "SongEditor3.dmg" \
                "dist/"
            print_status "DMG created: SongEditor3.dmg"
        else
            print_warning "create-dmg not found. Install with: brew install create-dmg"
        fi

    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        print_info "Windows distribution package created in dist/ folder"
        print_info "Consider using NSIS or Inno Setup for installer creation"

    else
        print_info "Linux distribution package created in dist/ folder"
        print_info "Consider using AppImage for better distribution"
    fi
}

# Test the executable
test_executable() {
    print_info "Testing executable..."

    # Create a simple test script
    cat > test_executable.py << 'EOF'
#!/usr/bin/env python3
"""
Simple test script to verify the executable works
"""

import sys
import os

def test_imports():
    """Test basic imports"""
    try:
        import PySide6.QtWidgets
        print("✅ PySide6 import successful")
    except ImportError as e:
        print(f"❌ PySide6 import failed: {e}")
        return False

    try:
        import torch
        print("✅ PyTorch import successful")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        import librosa
        print("✅ Librosa import successful")
    except ImportError as e:
        print(f"❌ Librosa import failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("🧪 Testing Song Editor 3 executable...")
    success = test_imports()
    if success:
        print("✅ All basic imports successful!")
        sys.exit(0)
    else:
        print("❌ Some imports failed!")
        sys.exit(1)
EOF

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -f "dist/Song Editor 3.app/Contents/MacOS/SongEditor3" ]; then
            print_info "Testing macOS executable..."
            python test_executable.py
        fi

    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        if [ -f "dist/SongEditor3.exe" ]; then
            print_info "Testing Windows executable..."
            python test_executable.py
        fi

    else
        if [ -f "dist/SongEditor3" ]; then
            print_info "Testing Linux executable..."
            python test_executable.py
        fi
    fi

    # Clean up test file
    rm -f test_executable.py
}

# Main build process
main() {
    echo ""
    print_info "Starting Song Editor 3 build process..."
    echo ""

    # Activate virtual environment
    source .venv/bin/activate

    # Run build steps
    check_prerequisites
    echo ""
    clean_build
    echo ""
    prepare_spec
    echo ""
    build_executable
    echo ""

    if verify_build; then
        echo ""
        create_distribution
        echo ""
        test_executable
        echo ""

        print_status "🎉 Build process completed successfully!"
        echo ""
        print_info "Next steps:"
        echo "1. Test the executable thoroughly on your system"
        echo "2. Test on other target platforms if cross-platform deployment"
        echo "3. Create distribution packages as needed"
        echo "4. Update version numbers and release notes"
        echo ""

        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "🚀 To run: open dist/Song\\ Editor\\ 3.app"
        elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
            echo "🚀 To run: dist/SongEditor3.exe"
        else
            echo "🚀 To run: ./dist/SongEditor3"
        fi

    else
        print_error "Build verification failed!"
        exit 1
    fi
}

# Run main function
main "$@"
