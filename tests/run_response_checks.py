# Standalone runner for environments without pytest; no native controller use.
import runpy
from pathlib import Path
for file in ['test_directional_sonar.py','test_matched_response.py']:
    tests=runpy.run_path(str(Path(__file__).parent/file))
    for name,value in sorted(tests.items()):
        if name.startswith('test_'):
            value()
            print('PASS',file,name)
