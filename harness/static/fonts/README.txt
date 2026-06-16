IBM Plex Sans woff2 vendor slot
================================

The web UI's CSS declares ``font-family: 'IBM Plex Sans', 'Segoe UI',
system-ui, ...`` and will pick up locally-installed IBM Plex Sans
automatically. To get the same typography everywhere (including
machines that don't have Plex installed), drop the OFL-1.1 woff2
files into this directory and uncomment the @font-face blocks in
``harness/static/css/app.css`` (search for ``@font-face``).

Recommended files (download from https://github.com/IBM/plex):
  - ibm-plex-sans-400.woff2  (regular weight)
  - ibm-plex-sans-600.woff2  (semi-bold)

License: SIL Open Font License 1.1. Add the LICENSE.txt from the
upstream Plex distribution to this directory when vendoring.

The system-font fallback stack is good — Segoe UI on Windows,
system-ui on macOS / Linux desktop. Operators in air-gapped, headless
environments may want the woff2s for visual consistency in screenshots
and shared dashboards.
