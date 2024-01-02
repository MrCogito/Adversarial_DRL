import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import Defaults, GPU

Defaults(name="2024v32_newscoresRand0", GPU=GPU.v32)
Defaults(name="2024v80_nrescoresRand0", GPU=GPU.a80)

