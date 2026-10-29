#!/usr/bin/env bash
# Install the handbook fonts (Source Serif 4, Fira Sans, IBM Plex Mono; SIL OFL) into ~/.local/share/fonts. Falls back to DejaVu if offline.
set -u; D="$HOME/.local/share/fonts/handbook"; mkdir -p "$D"
B=https://raw.githubusercontent.com/google/fonts/main/ofl
for w in Regular Italic Medium SemiBold Bold ExtraBold Black Light; do curl -fsSL -o "$D/FiraSans-$w.ttf" "$B/firasans/FiraSans-$w.ttf" || true; done
for w in Regular Medium SemiBold Bold Italic; do curl -fsSL -o "$D/IBMPlexMono-$w.ttf" "$B/ibmplexmono/IBMPlexMono-$w.ttf" || true; done
curl -fsSL -o "$D/SourceSerif4.ttf" "$B/sourceserif4/SourceSerif4%5Bopsz,wght%5D.ttf" || true; curl -fsSL -o "$D/SourceSerif4-Italic.ttf" "$B/sourceserif4/SourceSerif4-Italic%5Bopsz,wght%5D.ttf" || true
fc-cache -f >/dev/null 2>&1 || true; echo "fonts installed in $D"
