"""Export slide PNGs + PDF with PowerPoint and flag text that overflows its box."""
from pathlib import Path
import json, sys
import win32com.client
from PIL import Image

root = Path(__file__).resolve().parent
deck = next(root.glob('*.pptx'))
preview = root / 'preview'
preview.mkdir(exist_ok=True)
app = win32com.client.DispatchEx('PowerPoint.Application')
pres = app.Presentations.Open(str(deck), True, False, False)
issues = []
try:
    for slide in pres.Slides:
        slide.Export(str(preview / f'{slide.SlideIndex:02}.png'), 'PNG', 1600, 900)
        for shape in slide.Shapes:
            if shape.HasTextFrame and shape.TextFrame.HasText:
                tf = shape.TextFrame2
                if tf.TextRange.BoundHeight > shape.Height - tf.MarginTop - tf.MarginBottom + 1.5:
                    issues.append({'slide': slide.SlideIndex, 'text': shape.TextFrame.TextRange.Text[:50],
                                   'over_pt': round(tf.TextRange.BoundHeight - shape.Height, 1)})
    pres.SaveAs(str(deck.with_suffix('.pdf')), 32)
finally:
    pres.Close()
    app.Quit()
print(json.dumps({'slides': len(list(preview.glob('[0-9][0-9].png'))), 'overflow': issues}, ensure_ascii=False, indent=1))
