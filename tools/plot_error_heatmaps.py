"""Render signed/absolute error heatmaps from a native multimodel matrix.

The matrix is intentionally treated as data, rather than assuming the original
two-model/three-input layout.  This keeps the plots useful when additional
models (for example Qwen-27B) or input IDs are added.
"""
import argparse, json, math, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--absolute',action='store_true',help='render absolute errors (default is signed)')
    a=ap.parse_args()
    d=json.loads(a.matrix.read_text(encoding='utf-8'))
    # The frozen acceptance merger stores one aggregate per
    # model|prompt|output|parallel key.  Render that format directly so the
    # final 5x3x3x3 report does not silently produce an empty legacy plot.
    if d.get('aggregation') and any('|' in str(k) for k in d['aggregation']):
        render_aggregate_heatmaps(d, a.out_dir, a.absolute)
        return
    cells=[c for c in d.get('cells',[]) if c.get('status')=='valid']
    a.out_dir.mkdir(parents=True,exist_ok=True)
    models=list(d.get('models',{})) or sorted({c['model_key'] for c in cells})
    models=[m for m in models if any(c.get('model_key')==m for c in cells)]
    inputs=list(d.get('inputs',{})) or sorted({c['input_id'] for c in cells})
    runtimes=[]
    for c in cells:
        rid=c.get('runtime_id')
        if rid and rid not in runtimes: runtimes.append(rid)
    if not runtimes: runtimes=['cpu_only','partial','full']

    def model_label(key):
        meta=d.get('models',{}).get(key,{})
        geom=meta.get('geometry',{})
        arch=geom.get('architecture')
        layers=geom.get('n_layer')
        # Keep familiar names while retaining a useful geometry hint for new keys.
        known={'qwen25_0p5b':'Qwen2.5-0.5B','tinyllama_1p1b':'TinyLlama-1.1B',
               'qwen27b':'Qwen-27B','qwen2_7b':'Qwen-27B'}
        label=known.get(key, re.sub(r'[_-]+',' ',key).title())
        return f"{label} ({layers}L)" if layers else label

    # Build values once so the color scale is comparable across model panels.
    metric_values={}
    for metric in ('ttft_ms','tpot_ms','e2e_ms'):
        vals=[]
        for c in cells:
            v=(c.get('relative_error_pct') or {}).get(metric)
            if isinstance(v,(int,float)) and math.isfinite(v): vals.append(abs(float(v)) if a.absolute else float(v))
        metric_values[metric]=vals

    def nice_scale(values, signed):
        if not values: return 1.0
        top=max(abs(v) for v in values) if signed else max(values)
        if top<=0: return 1.0
        # Round up to a readable step while allowing >100% errors.
        step=25 if top<=200 else 50 if top<=500 else 100
        return max(step, math.ceil(top/step)*step)

    def color(v, vmax, signed):
        if v is None: return (224,224,224)
        if signed:
            t=max(-1.0,min(1.0,float(v)/vmax)); q=abs(t)
            if t>=0: return (255, int(245-145*q), int(245-145*q))
            return (int(245-145*q), int(245-145*q), 255)
        t=max(0.0,min(1.0,float(v)/vmax))
        return (255, int(255-150*t), int(220-170*t))

    signed=not a.absolute
    for metric in ('ttft_ms','tpot_ms','e2e_ms'):
        panel_h=170; width=1080; height=panel_h*len(models)+110
        img=Image.new('RGB',(width,height),'white'); draw=ImageDraw.Draw(img)
        title=('Absolute ' if a.absolute else 'Signed ')+metric.replace('_',' ')+' error (%)'
        draw.text((20,15),title,fill='black')
        vmax=nice_scale(metric_values[metric],signed)
        for i,model in enumerate(models):
            vals=[[None]*len(runtimes) for _ in inputs]
            for c in cells:
                if c['model_key']==model and c['input_id'] in inputs and c['runtime_id'] in runtimes:
                    v=(c.get('relative_error_pct') or {}).get(metric)
                    if isinstance(v,(int,float)) and math.isfinite(v):
                        vals[inputs.index(c['input_id'])][runtimes.index(c['runtime_id'])]=abs(float(v)) if a.absolute else float(v)
            y0=55+i*panel_h
            draw.text((20,y0),model_label(model),fill='black')
            cell_w=250; left=165; row_h=34; gap=8
            for col,rt in enumerate(runtimes): draw.text((left+col*cell_w,y0),str(rt),fill='black')
            for r,input_id in enumerate(inputs):
                y=y0+22+r*(row_h+gap); draw.text((20,y+9),str(input_id),fill='black')
                for col in range(len(runtimes)):
                    x=left+col*cell_w; v=vals[r][col]; txt='NA' if v is None else f'{v:.1f}%'
                    fill=color(v,vmax,signed)
                    draw.rectangle((x,y,x+cell_w-14,y+row_h),fill=fill,outline='black')
                    lum=(0.299*fill[0]+0.587*fill[1]+0.114*fill[2]); ink='white' if lum<145 else 'black'
                    draw.text((x+(cell_w-14)//2-18,y+9),txt,fill=ink)
        # compact scale legend shared by all model panels
        lx,ly=left,height-28; segments=12; sw=45
        for j in range(segments):
            if signed: lv=-vmax+2*vmax*j/(segments-1)
            else: lv=vmax*j/(segments-1)
            draw.rectangle((lx+j*sw,ly,lx+(j+1)*sw,ly+12),fill=color(lv,vmax,signed),outline=None)
        if signed:
            draw.text((lx,ly+14),f'-{vmax:g}%',fill='black'); draw.text((lx+segments*sw//2-12,ly+14),'0%',fill='black'); draw.text((lx+segments*sw-34,ly+14),f'+{vmax:g}%',fill='black')
        else:
            draw.text((lx,ly+14),'0%',fill='black'); draw.text((lx+segments*sw-34,ly+14),f'{vmax:g}%',fill='black')
        out=a.out_dir/(f"{metric}_{'abs' if a.absolute else 'signed'}_heatmap.png")
        img.save(out); print(out)
def render_aggregate_heatmaps(d, out_dir, absolute):
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(d.get('models') or [])
    keys = []
    for key in d.get('aggregation', {}):
        parts = str(key).split('|')
        if len(parts) == 4 and parts not in keys: keys.append(parts)
    keys.sort(key=lambda x: (x[0], x[1], x[2], int(x[3])))
    scenarios = sorted({(p, o, n) for _, p, o, n in keys}, key=lambda x: (x[0], x[1], int(x[2])))
    labels = [f'{p}/{o}/p{n}' for p, o, n in scenarios]
    for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms'):
        field = 'median_abs_pct' if absolute else 'median_signed_pct'
        values = []
        for model in models:
            row = []
            for p, o, n in scenarios:
                g = d['aggregation'].get(f'{model}|{p}|{o}|{n}', {})
                v = ((g.get('metrics') or {}).get(metric) or {}).get(field)
                row.append(float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None)
            values.append(row); values_flat = [v for r in values for v in r if v is not None]
        signed = not absolute
        top = max((abs(v) for v in values_flat), default=1.0) if signed else max(values_flat, default=1.0)
        vmax = max(25.0, math.ceil(top / 25.0) * 25.0)
        width = max(1200, 170 + 112 * len(scenarios)); height = 90 + 48 * len(models)
        img = Image.new('RGB', (width, height), 'white'); draw = ImageDraw.Draw(img)
        title = ('Absolute ' if absolute else 'Signed ') + metric.replace('_', ' ') + ' median error (%)'
        draw.text((10, 8), title, fill='black'); left, top_y = 165, 35; cw, rh = 108, 32
        for j, label in enumerate(labels): draw.text((left+j*cw, top_y), label, fill='black')
        for i, model in enumerate(models):
            y = top_y + 20 + i*rh; draw.text((10, y+8), str(model), fill='black')
            for j, v in enumerate(values[i]):
                x = left + j*cw; fill = (224,224,224) if v is None else aggregate_color(v, vmax, signed)
                draw.rectangle((x, y, x+cw-3, y+rh-3), fill=fill, outline='black')
                txt = 'NA' if v is None else f'{v:.1f}%'; draw.text((x+4, y+8), txt, fill='black')
        path = out_dir / f'{metric}_{"abs" if absolute else "signed"}_heatmap.png'; img.save(path); print(path)


def aggregate_color(v, vmax, signed):
    if signed:
        t = max(-1.0, min(1.0, float(v)/vmax)); q = abs(t)
        return (255, int(245-145*q), int(245-145*q)) if t >= 0 else (int(245-145*q), int(245-145*q), 255)
    t = max(0.0, min(1.0, float(v)/vmax)); return (255, int(255-150*t), int(220-170*t))


if __name__=='__main__': main()
