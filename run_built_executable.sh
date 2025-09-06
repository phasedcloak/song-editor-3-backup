#!/bin/bash

# Runtime wrapper for Song Editor 3 executable
# Fixes dyld library loading issues on macOS

echo "🚀 Running Song Editor 3 with dyld fixes..."

# Set environment variables to fix dyld issues
export DYLD_LIBRARY_PATH="/usr/lib:/System/Library/Frameworks:/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export DYLD_FRAMEWORK_PATH="/System/Library/Frameworks:/opt/homebrew/lib:$DYLD_FRAMEWORK_PATH"
export DYLD_FALLBACK_LIBRARY_PATH="/usr/lib:/usr/local/lib:/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"
export DYLD_FALLBACK_FRAMEWORK_PATH="/System/Library/Frameworks:/opt/homebrew/lib:$DYLD_FALLBACK_FRAMEWORK_PATH"

# Disable library validation (helps with some macOS versions)
export DYLD_IGNORE_MISSING_LIBRARIES=1

# Set Python environment
export PYTHONPATH="/opt/homebrew/lib/python3.10/site-packages:$PYTHONPATH"

# Run the executable with all arguments passed through
echo "📋 Command: ./dist/SongEditor3 $@"
echo ""

./dist/SongEditor3 "$@"

echo ""
echo "✅ Execution completed"
