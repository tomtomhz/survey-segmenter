"""Desktop-app entry point for Survey Segmenter (this is what the packaged .app runs).

It just starts the local web app. Set the SEG_PORT environment variable to force a specific port
(used only for automated testing of the built binary); normally a free port is chosen automatically.
"""
import os

import segment_kmeans
import webapp              # the web application itself
import kprototypes         # the mixed-question engine, reached through charts.py lazily
import clusterability      # the second cluster-tendency test, and its compiled dip extension
import ai_interpret        # the optional Claude chat layer

# Imported here by name, not only reached through segment_kmeans, because a bundler follows
# imports statically. segment_kmeans.serve() imports webapp lazily to avoid a cycle, and a lazy
# import is invisible to PyInstaller — an app packaged without its own web server is exactly the
# failure that shipped once already when matplotlib's SVG backend was left out.

if __name__ == "__main__":
    _ = (ai_interpret, webapp, kprototypes, clusterability)   # kept for linters and bundlers
    _port = os.environ.get("SEG_PORT")
    if _port:
        segment_kmeans.serve(int(_port))
    else:
        segment_kmeans.app()
