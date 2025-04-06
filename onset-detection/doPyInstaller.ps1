Remove-Item build/ -ErrorAction Ignore -Recurse
Remove-Item dist/ -ErrorAction Ignore -Recurse
Remove-Item audiostellar-onset-detection.spec -ErrorAction Ignore -Recurse
pyinstaller --onefile --name audiostellar-onset-detection --additional-hooks-dir=. onset-detection.py
cp dist/audiostellar-onset-detection.exe ../../bin/
echo "Binary has been built and copied to ../../bin/ successfully."
