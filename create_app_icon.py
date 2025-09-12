#!/usr/bin/env python3
"""
Create Application Icon Script

This script helps create the application icon files from the provided image.
The image description shows a neon-style microphone and musical note logo with
"Sweet Transcribe" text on a dark background.

To use this script:
1. Save the provided image as 'sweet_transcribe_logo.png' in the project root
2. Run this script to create the appropriate icon files
3. The script will create icon.png, icon.ico, and icon.icns files
"""

import os
import sys
from pathlib import Path

def create_icon_files():
    """Create icon files from the source image."""
    project_root = Path(__file__).parent
    source_image = project_root / "sweet_transcribe_logo.png"
    
    if not source_image.exists():
        print("❌ Source image 'sweet_transcribe_logo.png' not found!")
        print("Please save the provided image as 'sweet_transcribe_logo.png' in the project root.")
        return False
    
    print("🎨 Creating application icons from sweet_transcribe_logo.png...")
    
    try:
        from PIL import Image
        
        # Load the source image
        img = Image.open(source_image)
        print(f"✅ Loaded source image: {img.size}")
        
        # Create different sizes for different platforms
        sizes = {
            "icon.png": 512,  # Linux
            "icon_256.png": 256,  # For ICO
            "icon_48.png": 48,    # For ICO
            "icon_32.png": 32,    # For ICO
            "icon_16.png": 16,    # For ICO
        }
        
        # Create PNG files
        for filename, size in sizes.items():
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            output_path = project_root / filename
            resized.save(output_path, "PNG")
            print(f"✅ Created {filename} ({size}x{size})")
        
        # Create ICO file (Windows)
        try:
            ico_sizes = [(16, 16), (32, 32), (48, 48), (256, 256)]
            ico_images = []
            for size in ico_sizes:
                resized = img.resize(size, Image.Resampling.LANCZOS)
                ico_images.append(resized)
            
            ico_path = project_root / "icon.ico"
            ico_images[0].save(ico_path, format='ICO', sizes=ico_sizes)
            print(f"✅ Created icon.ico (Windows)")
        except Exception as e:
            print(f"⚠️  Could not create ICO file: {e}")
        
        # Create ICNS file (macOS) - requires iconutil
        try:
            # Create iconset directory
            iconset_dir = project_root / "icon.iconset"
            iconset_dir.mkdir(exist_ok=True)
            
            # Create different sizes for ICNS
            icns_sizes = {
                "icon_16x16.png": 16,
                "icon_32x32.png": 32,
                "icon_128x128.png": 128,
                "icon_256x256.png": 256,
                "icon_512x512.png": 512,
                "icon_1024x1024.png": 1024,
            }
            
            for filename, size in icns_sizes.items():
                resized = img.resize((size, size), Image.Resampling.LANCZOS)
                output_path = iconset_dir / filename
                resized.save(output_path, "PNG")
            
            # Convert to ICNS using iconutil
            import subprocess
            result = subprocess.run([
                "iconutil", "-c", "icns", str(iconset_dir), "-o", str(project_root / "icon.icns")
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Created icon.icns (macOS)")
            else:
                print(f"⚠️  Could not create ICNS file: {result.stderr}")
            
            # Clean up iconset directory
            import shutil
            shutil.rmtree(iconset_dir)
            
        except Exception as e:
            print(f"⚠️  Could not create ICNS file: {e}")
        
        print("\n🎉 Icon creation completed!")
        print("The following files have been created:")
        print("  - icon.png (512x512) - Linux/fallback")
        print("  - icon.ico - Windows")
        print("  - icon.icns - macOS")
        print("\nThe application will now use these icons automatically.")
        
        return True
        
    except ImportError:
        print("❌ PIL (Pillow) not installed!")
        print("Install it with: pip install Pillow")
        return False
    except Exception as e:
        print(f"❌ Error creating icons: {e}")
        return False

if __name__ == "__main__":
    print("🎵 Sweet Transcribe - Icon Creator")
    print("=" * 40)
    
    if create_icon_files():
        print("\n✅ All done! Your application now has a custom icon.")
    else:
        print("\n❌ Icon creation failed. Please check the errors above.")
