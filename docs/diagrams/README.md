# Diagrams

The README's diagrams are written in Mermaid (`*.mmd`) and committed as rendered PNGs. GitHub draws Mermaid text in
the viewer's own font, which can be wider than the font Mermaid sized the boxes with, so labels get clipped. A PNG
looks the same everywhere.

After editing a `.mmd` file, re-render it (Node 18+; Mermaid CLI 11.4.2; any Chromium-based browser):

```sh
npx -y @mermaid-js/mermaid-cli@11.4.2 -i docs/diagrams/request_path.mmd -o docs/diagrams/request_path.png -b white -s 2
```

`-s 2` renders at twice the resolution, so the image stays sharp on high-DPI screens. The first line of each source
fixes the font (Arial), the padding and plain SVG text, so the rendering is reproducible.
