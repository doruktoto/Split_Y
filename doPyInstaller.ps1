Remove-Item build/ -ErrorAction Ignore -Recurse
Remove-Item dist/ -ErrorAction Ignore -Recurse
Remove-Item audiostellar-data-analysis.spec -ErrorAction Ignore -Recurse
pyinstaller --onefile --name audiostellar-data-analysis --additional-hooks-dir=. doProcess.py
cp dist/audiostellar-data-analysis.exe ../bin/
echo "Binary has been built and copied to ../bin/ successfully."
