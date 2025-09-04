#!/bin/bash

# Prepare OpenAI Whisper Debug Project for GitHub

echo "=== Preparing OpenAI Whisper Debug Project for GitHub ==="
echo

# Check if we're in the right directory
if [ ! -f "standalone_openai_whisper_test.py" ]; then
    echo "❌ Error: Please run this script from the Song_Editor_3 directory"
    exit 1
fi

# Create a clean directory for GitHub
GITHUB_DIR="openai_whisper_debug_project"
echo "📁 Creating clean directory: $GITHUB_DIR"
rm -rf "$GITHUB_DIR"
mkdir -p "$GITHUB_DIR"

# Copy all test files
echo "📋 Copying test files..."
cp standalone_openai_whisper_test.py "$GITHUB_DIR/"
cp standalone_openai_whisper_test_fixed.py "$GITHUB_DIR/"
cp test_wav_to_karaoke_exact_copy.py "$GITHUB_DIR/"
cp debug_resource_usage.py "$GITHUB_DIR/"
cp WHISPER_COMPARISON_ANALYSIS.md "$GITHUB_DIR/"
cp OPENAI_WHISPER_DEBUG_README.md "$GITHUB_DIR/"

# Copy audio file if it exists
if [ -f "25-03-12 we see your love - 02.wav" ]; then
    echo "🎵 Copying audio file..."
    cp "25-03-12 we see your love - 02.wav" "$GITHUB_DIR/"
else
    echo "⚠️  Warning: Audio file not found"
fi

# Create requirements.txt for the debug project
echo "📦 Creating requirements.txt..."
cat > "$GITHUB_DIR/requirements.txt" << EOF
# OpenAI Whisper Debug Project Requirements
openai-whisper>=20231117
numpy>=1.26.0
soundfile>=0.12.0
psutil>=5.9.0
torch>=2.0.0
torchaudio>=2.0.0
EOF

# Create a simple setup script
echo "🔧 Creating setup script..."
cat > "$GITHUB_DIR/setup.sh" << 'EOF'
#!/bin/bash

# Setup script for OpenAI Whisper Debug Project

echo "=== Setting up OpenAI Whisper Debug Environment ==="

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo "Run: source .venv/bin/activate"
echo "Then test with: python standalone_openai_whisper_test_fixed.py"
EOF

chmod +x "$GITHUB_DIR/setup.sh"

# Create .gitignore
echo "🚫 Creating .gitignore..."
cat > "$GITHUB_DIR/.gitignore" << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
memory_usage.log

# Temporary files
temp/
tmp/
EOF

# Create a quick test script
echo "🧪 Creating quick test script..."
cat > "$GITHUB_DIR/quick_test.py" << 'EOF'
#!/usr/bin/env python3
"""
Quick test to verify OpenAI Whisper installation
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import whisper
    print("✅ OpenAI Whisper imported successfully")
    print(f"   Version: {whisper.__version__}")
    
    # Test model loading
    print("\n🔧 Testing model loading...")
    model = whisper.load_model("tiny")
    print("✅ Model loaded successfully")
    
    print("\n🎯 OpenAI Whisper is working! Ready for debugging.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
EOF

# Make scripts executable
chmod +x "$GITHUB_DIR/quick_test.py"

# Create summary
echo
echo "✅ GitHub project prepared successfully!"
echo
echo "📁 Directory: $GITHUB_DIR"
echo "📋 Files included:"
echo "   - All test scripts"
echo "   - Audio file (if found)"
echo "   - Comprehensive README"
echo "   - Requirements and setup scripts"
echo "   - .gitignore for Python projects"
echo
echo "🚀 Next steps:"
echo "   1. cd $GITHUB_DIR"
echo "   2. git init"
echo "   3. git add ."
echo "   4. git commit -m 'Initial commit: OpenAI Whisper debugging project'"
echo "   5. Create new repository on GitHub"
echo "   6. git remote add origin <your-github-repo-url>"
echo "   7. git push -u origin main"
echo
echo "📖 The README file contains detailed instructions for reproducing the issue."
echo "🔍 Use debug_resource_usage.py to monitor resources during the hang."
