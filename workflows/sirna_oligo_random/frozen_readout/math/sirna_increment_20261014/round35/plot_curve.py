"""Plot already verified aggregate values; no new model or metric selection."""
import os,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(R/'mplconfig')
os.environ['OMP_NUM_THREADS']='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    d=json.loads((R/'diagnostic.json').read_text(encoding='utf-8'))
    v=json.loads((R/'verification.json').read_text(encoding='utf-8'))
    assert v['passed'] and v['diagnostic_sha256']==hashlib.sha256((R/'diagnostic.json').read_bytes()).hexdigest()
    assert not (R/'learning_curve.png').exists() and not (R/'learning_curve.svg').exists()
    fig,axes=plt.subplots(1,3,figsize=(11,3.65),sharey=True)
    for ax,setting in zip(axes,('Huesken','Mixset','Takayuki')):
        for condition,label,color in (('core_real','Core','#0072B2'),('full_real','Full','#D55E00'),('initial_core','Exact initial','#707070')):
            rows=sorted((a for a in d['aggregate'] if a['setting']==setting and a['condition']==condition),key=lambda a:a['fraction'])
            x=[100*a['fraction'] for a in rows];y=[a['PCC_delta_mean'] for a in rows]
            ax.plot(x,y,color=color,label=label,lw=1.6)
            ax.errorbar(x[:2],y[:2],yerr=[a['PCC_delta_SD'] for a in rows[:2]],fmt='o',color=color,capsize=3,ms=4,lw=1)
            ax.plot(x[2],y[2],marker='D',color=color,ms=4)
        ax.axhline(0,color='#444444',lw=.8,ls='--')
        ax.set_title(setting,fontsize=11);ax.set_xticks([50,75,100]);ax.set_xlabel('Training subset (%)');ax.grid(alpha=.15)
        ax.spines[['right','top']].set_visible(False)
    axes[0].set_ylabel('PCC difference vs matched TD24')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.985),frameon=False)
    fig.text(.5,.025,'Internal validation only. 50/75%: mean and SD of 3 subset draws; 100%: one reused reference, uncertainty not estimated.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.78,wspace=.14)
    fig.savefig(R/'learning_curve.png',dpi=180);fig.savefig(R/'learning_curve.svg');plt.close(fig)
    print('PLOT_SAVED')


if __name__=='__main__':main()
