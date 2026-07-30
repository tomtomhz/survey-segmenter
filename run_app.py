"""Desktop-app entry point for Survey Segmenter (this is what the packaged .app runs).

It just starts the local web app. Set the SEG_PORT environment variable to force a specific port
(used only for automated testing of the built binary); normally a free port is chosen automatically.
"""
import os

import segment_kmeans
import ai_interpret        # imported so the packaged app bundles the optional Claude chat layer

if __name__ == "__main__":
    _ = ai_interpret        # reference it so linters/bundlers treat the import as used
    _port = os.environ.get("SEG_PORT")
    if _port:
        segment_kmeans.serve(int(_port))
    else:
        segment_kmeans.app()
