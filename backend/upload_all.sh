#!/bin/bash
# backend/upload_all.sh

ENDPOINT="http://localhost:8000/api/documents/upload"
UPLOAD_DIR="data/uploads"

echo "=== Processing files in $UPLOAD_DIR ==="

for file in "$UPLOAD_DIR"/*; do
    # Skip directories and UUID-prefixed files
    if [ -f "$file" ] && [[ $(basename "$file") != *-*-*-*-* ]]; then
        echo "Uploading $(basename "$file")..."
        
        # Upload via API
        response=$(curl -s -X POST "$ENDPOINT" \
            -F "files=@$file" \
            -H "Content-Type: multipart/form-data")
        
        # Check response
        if [[ $response == *"processing"* ]]; then
            echo "✓ Successfully submitted $(basename "$file")"
        else
            echo "✗ Failed to upload $(basename "$file"): $response"
        fi
        
        # Rate limiting to prevent overloading
        sleep 2
    fi
done

echo "=== Upload process complete ==="