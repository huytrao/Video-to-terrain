import sys
sys.path.insert(0, '/Users/traoanhuy/source_code/Full-pipeline-scan-terrain/Videos_to_terrain/src')
from v2t.config import CFG
print('CFG PROJECT_ROOT:', CFG.run_dir)
print('CFG.paths.simulation:', CFG.paths.simulation)
print('CFG.paths.collision:', CFG.paths.collision)
print('module file:', __file__)
import json, os
os.makedirs(CFG.paths.collision / 'walls', exist_ok=True)
p = CFG.paths.collision / 'floor_plane.json'
print('Writing to:', p)
with open(p, 'w') as f:
    json.dump({'type': 'plane'}, f)
print('Written:', p, os.path.exists(p))
