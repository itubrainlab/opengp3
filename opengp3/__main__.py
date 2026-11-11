"""``python -m opengp3``: same as the ``opengp3`` command (default: the server)."""
import sys

from .cli import main

sys.exit(main())
