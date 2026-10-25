"""Regenerate the current Figure 1 and its traceable source-data exports.

The current icon-led, multicolour layout is implemented in
fig1_redesign_v2.py. Earlier renderings and sources are preserved under
figures/revisions/.
"""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('fig1_redesign_v2.py')), run_name='__main__')
