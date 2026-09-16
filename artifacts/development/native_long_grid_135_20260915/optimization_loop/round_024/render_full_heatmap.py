"""Presentation-only rendering of immutable R24 engine error scores; no scoring/refit."""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
ROOT=Path(__file__).resolve().parent
METRICS=('engine_ttft_ms','engine_tpot_ms','engine_e2e_ms')
LABELS=('Engine TTFT','Engine TPOT','Engine E2E')
COLORS=LinearSegmentedColormap.from_list('error',[(0,'#e8f4ef'),(.09999,'#6bb69a'),(.1,'#fce4a3'),(.2,'#f3b15e'),(.5,'#d76c5e'),(1,'#913446')]);COLORS.set_bad('#e4e7eb')

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--score',type=Path,default=ROOT/'repaired/errors.0001.json');a.add_argument('--output',type=Path,default=ROOT/'full_engine_error_heatmap.png');args=a.parse_args()
 raw=args.score.read_bytes();doc=json.loads(raw);assert doc['selected_denominator']==131 and len(doc['cells'])==131 and doc['blind_evaluation'] is False
 targets=[args.output,args.output.with_suffix('.svg'),args.output.with_suffix('.provenance.json')];assert not any(x.exists() for x in targets),'refusing overwrite'
 groups=sorted({r['model_key'] for r in doc['cells']});fig,axes=plt.subplots(len(groups),3,figsize=(18,3.15*len(groups)),squeeze=False);norm=Normalize(0,100)
 for row_index,group in enumerate(groups):
  cells=[r for r in doc['cells'] if r['model_key']==group]
  for col,metric in enumerate(METRICS):
   ax=axes[row_index,col];values=np.full((3,9),np.nan);texts={}
   for r in cells:
    match=re.search(r'_p(128|512|1536)_o(32|128|256)_c(1|2|4)(?:__|$)',r['cell_id']);assert match,r['cell_id']
    prompt,output,parallel=map(int,match.groups());y=(1,2,4).index(parallel);x=(128,512,1536).index(prompt)*3+(32,128,256).index(output);m=r['metrics'][metric]
    if m['status']=='scored':
     values[y,x]=float(m['absolute_percentage_error_pct']);texts[y,x]=('<10' if values[y,x]<10 and round(values[y,x],1)==10 else f"{values[y,x]:.1f}")
    else:texts[y,x]='X'
   im=ax.imshow(values,cmap=COLORS,norm=norm,aspect='auto')
   for y in range(3):
    for x in range(9):
     value=values[y,x];ax.text(x,y,texts.get((y,x),'-'),ha='center',va='center',fontsize=8,color='white' if np.isfinite(value) and value>45 else '#24313d')
   ax.set_xticks(range(9),[f'{p}/{o}' for p in (128,512,1536) for o in (32,128,256)],rotation=55,ha='right',fontsize=8);ax.set_yticks(range(3),['C1','C2','C4']);ax.set_title(group+' | '+LABELS[col],loc='left',fontsize=11,fontweight='bold');ax.set_xlabel('Prompt / Output tokens',fontsize=8)
   ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True);ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',bottom=False,left=False)
 fig.suptitle('R24 retained proof identity repair | Fixed native development set | Engine absolute relative error (%)',fontsize=15,y=.995)
 fig.subplots_adjust(left=.045,right=.94,top=.965,bottom=.105,hspace=.78,wspace=.2)
 cax=fig.add_axes([.955,.15,.013,.7]);fig.colorbar(im,cax=cax,label='Absolute error (%) | strict target <10')
 fig.text(.045,.012,'Fixed set: 131 cells from 162 planned cells.  - = outside fixed set; X = failed/unscored.\nRetained KV treatment: 62 ordinary-attention candidates; 69 hybrid cases preserve prior model fallback. Independent acceptance (B): unvalidated. All failures remain in denominators.',fontsize=10,color='#354557')
 args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=170,facecolor='white');fig.savefig(args.output.with_suffix('.svg'),facecolor='white');plt.close(fig)
 assert args.score.read_bytes()==raw,'source changed during rendering'
 receipt={'schema':'presentation-only-error-heatmap/v1','source':{'path':str(args.score.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)},'changes_scores_or_acceptance':False,'figure_denominator':131,'planned_native_grid':162,'outputs':[str(x) for x in targets[:2]]}
 with targets[2].open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(receipt))
if __name__=='__main__':main()
