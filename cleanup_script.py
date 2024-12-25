import os
from pathlib import Path
import re

def rename_photos(directory='.'):
    # Get all PXL files in directory
    pattern = re.compile(r'PXL_\d{8}_\d+(?:\.MP)?\.jpg')
    photos = [f for f in os.listdir(directory) if pattern.match(f)]
    
    # Sort by extracting timestamp parts and comparing them
    def get_timestamp(filename):
        # Extract date and time parts, ignoring .MP suffix
        parts = filename.split('_')
        return (parts[1], parts[2].split('.')[0])
    
    photos.sort(key=get_timestamp)
    
    # Rename files
    for index, photo in enumerate(photos, 1):
        old_path = Path(directory) / photo
        new_path = old_path.parent / f"{index}.jpg"
        
        while new_path.exists():
            new_path = Path(str(new_path.parent / new_path.stem) + '_dup.jpg')
            
        os.rename(old_path, new_path)
        print(f"Renamed: {photo} → {new_path.name}")

if __name__ == '__main__':
    rename_photos()