# Vendored third-party JS

Vendored (not CDN-loaded) so the dashboard works offline on a loopback laptop
and to keep the CDN attack surface minimal — see `_safe_cdn_url` in
`harness/dashboard.py`.

## htmx 1.9.12

- Source: https://github.com/bigskysoftware/htmx (v1.9.12)
- File: `htmx-1.9.12.min.js` (48 KB, minified)
- License: BSD 2-Clause "Simplified"
- Copyright (c) 2020-2024, Big Sky Software

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## htmx SSE extension 1.9.12

- Source: https://github.com/bigskysoftware/htmx/tree/master/src/ext (v1.9.12)
- File: `htmx-sse-1.9.12.min.js` (10 KB)
- License: BSD 2-Clause "Simplified" (same as htmx above)

Required by the activity-feed + cost-meter (Phase 2) — enables
`hx-ext="sse" sse-connect="…"` on `<aside>` blocks to stream events into
Alpine stores without a page reload.

## Alpine.js 3.14.1

- Source: https://github.com/alpinejs/alpine (v3.14.1)
- File: `alpine-3.14.1.min.js` (44 KB, minified CDN bundle)
- License: MIT
- Copyright (c) 2019-2024 Caleb Porzio and contributors

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Refreshing versions

The three files above are pinned. To bump, replace with an equal-license
successor from the same upstream, keep the version in the filename, and
update this file. Do NOT switch to CDN loading — that reintroduces the
attack surface `_safe_cdn_url` exists to bound.
