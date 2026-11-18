"""Plots only for the frozen retention comparison; never chooses cuts."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scripts.compare_pilot_retentions import FRACTIONS, PROBABILITIES, GRID_PROBABILITIES, fraction_name, target_name, subset
from scripts.study_tail_pilot_cuts import qcd, finish, visible_range, density_hist, score_axes
from scripts.plot_ae_tail_eval_suite import weighted_kde, PALETTE

BRANCHES = ('full','tail')
COLORS = {'full':'tab:blue','tail':'tab:green'}
LABELS = ['100%','20%','10%']


def render(plan, results, calibration, checking, grids, checking_selection, names, output, kde_grid=64,
           branches=BRANCHES, fractions=FRACTIONS, probabilities=PROBABILITIES):
    # Optional dimensions let the double-NURD experiment reuse exactly the same
    # weighted scientific plots without changing historical study defaults.
    BRANCHES, FRACTIONS, PROBABILITIES = branches, fractions, probabilities
    COLORS = dict(zip(branches, ('tab:blue', 'tab:green')))
    LABELS = [f'{f:.0%}' for f in fractions]
    positions = range(len(fractions))
    n_branches, n_fractions = len(branches), len(fractions)
    output = Path(output); output.mkdir()
    cuts = plan['AE_thresholds']; minimums = plan['minimum_region_ESS']
    primary = target_name(plan['primary_overall_QCD_A_target'])
    n_targets = len(plan['overall_QCD_A_targets'])
    fig, axes = plt.subplots(2, n_targets, figsize=(6*n_targets,9), squeeze=False)
    for row,role in enumerate(('calibration_candidate','checking_point')):
        for col,target in enumerate(plan['overall_QCD_A_targets']):
            ax = axes[row,col]
            for branch in BRANCHES:
                values = [results[branch][fraction_name(f)][target_name(target)][role] for f in FRACTIONS]
                ys = [v['predicted_over_observed'] if v is not None else np.nan for v in values]
                ax.plot(positions, ys, color=COLORS[branch], alpha=.4)
                for x,value in enumerate(values):
                    if value is None: continue
                    if np.isfinite(value['predicted_over_observed']) and np.isfinite(value['ratio_uncertainty']):
                        ax.errorbar(x,value['predicted_over_observed'],yerr=value['ratio_uncertainty'],
                                    fmt='o' if value['statistically_valid'] else 'x',color=COLORS[branch])
                ax.plot([],[],color=COLORS[branch],marker='o',label=branch+' model')
            ax.axhline(1,c='black',ls='--');ax.set_xticks(positions,LABELS)
            ax.set(xlabel='Retained physical mixture',ylabel='Predicted / observed QCD',
                   title=f'{"Calibration (tuned)" if row==0 else "Checking (fixed cuts)"}; target A/N={target:.1%}')
            ax.legend(fontsize=9)
    finish(fig,output/'matched_overall_acceptance_closure.png')

    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for role,selection,color in (('calibration',plan['calibration_selection'],'tab:blue'),('checking',checking_selection,'tab:orange')):
        axes[0].plot(positions,[selection[fraction_name(f)]['all_backgrounds']['retention']*100 for f in FRACTIONS],
                     'o-',color=color,label=role)
    axes[0].set(ylabel='Retained physical mixture (%)',xlabel='Requested retention');axes[0].legend()
    for branch in BRANCHES:
        for role,style in (('calibration_candidate','--'),('checking_point','-')):
            values=[results[branch][fraction_name(f)][primary][role] for f in FRACTIONS]
            axes[1].plot(positions,[np.nan if v is None else v['A_efficiency_overall_QCD']*100 for v in values],
                         marker='o',ls=style,color=COLORS[branch],label=f'{branch}: '+('calibration' if role.startswith('cal') else 'checking'))
    axes[1].axhline(plan['primary_overall_QCD_A_target']*100,c='black',ls=':')
    axes[1].set(ylabel='A / original QCD yield (%)',xlabel='Retained physical mixture');axes[1].legend(fontsize=8)
    for ax in axes: ax.set_xticks(positions,LABELS)
    finish(fig,output/'acceptance_transfer.png')

    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for role,selection,color in (('calibration',plan['calibration_selection'],'tab:blue'),('checking',checking_selection,'tab:orange')):
        axes[0,0].plot(positions,[selection[fraction_name(f)]['QCD']['selected_effective_events'] for f in FRACTIONS],
                       'o-',color=color,label=role)
        axes[0,1].plot(positions,[selection[fraction_name(f)]['QCD']['events']*selection[fraction_name(f)]['QCD']['raw_retention'] for f in FRACTIONS],
                       'o-',color=color,label=role)
    axes[0,0].set(ylabel='Selected QCD effective events');axes[0,1].set(ylabel='Selected QCD raw rows')
    for ax in axes[0]: ax.set_xticks(positions,LABELS);ax.legend()
    for ax,role in zip(axes[1],('calibration_candidate','checking_point')):
        for k,(branch,f) in enumerate((b,f) for b in BRANCHES for f in FRACTIONS):
            value=results[branch][fraction_name(f)][primary][role]
            if value is not None:
                ax.bar(np.arange(4)+(k-(n_branches*n_fractions-1)/2)*.12,[value['regions'][r]['effective_count'] for r in 'ABCD'],
                       width=.12,label=f'{branch}, {f:.0%}')
        ax.scatter(range(4),minimums,marker='_',s=400,c='black',label='ESS requirement')
        ax.set_xticks(range(4),list('ABCD'));ax.set(yscale='log',ylabel='Region effective events',title=role)
        ax.legend(fontsize=7)
    finish(fig,output/'effective_statistics.png')

    for role in ('calibration','checking'):
        fig,axes=plt.subplots(n_branches,n_fractions,figsize=(6*n_fractions,5*n_branches),squeeze=False)
        for row,branch in enumerate(BRANCHES):
            for col,f in enumerate(FRACTIONS):
                key=fraction_name(f);grid=grids[branch][key][role];ax=axes[row,col]
                values=np.where(grid['valid'],abs(grid['ratio']-1)*100,np.nan)
                mesh=ax.pcolormesh(GRID_PROBABILITIES,GRID_PROBABILITIES,values.T,shading='auto',vmin=0,vmax=100,cmap='viridis_r')
                fig.colorbar(mesh,ax=ax,label='|Predicted/observed - 1| (%)',extend='max')
                candidate=plan['plans'][branch][key][primary]['candidate']
                if candidate is not None:
                    c=subset(calibration[branch],cuts[key]);w=c['weight']
                    x=w[c['ae']<=candidate['ae_cut']].sum()/w.sum()
                    y=w[c['md']<=candidate['md_cut']].sum()/w.sum()
                    ax.plot(x,y,'r*',ms=13)
                if not grid['valid'].any():ax.text(.5,.5,'No valid cells',ha='center',transform=ax.transAxes)
                ax.set(xlabel='Calibration-selected AE percentile',ylabel='Calibration-selected MD percentile',
                       title=f'{branch}, {f:.0%}: '+('diagnostic only' if role=='checking' else 'descriptive grid'))
        finish(fig,output/f'{role}_closure_scans.png')

    fig,axes=plt.subplots(n_branches,n_fractions,figsize=(6*n_fractions,5*n_branches),squeeze=False)
    for row,branch in enumerate(BRANCHES):
        for col,f in enumerate(FRACTIONS):
            key=fraction_name(f);ax=axes[row,col]
            entry=results[branch][key][primary]
            for role,curve,color in [('calibration',plan['plans'][branch][key][primary]['curve'],'tab:blue'),
                                     ('checking',entry['checking_curve'],'tab:orange')]:
                y=np.array([np.nan if p is None else p['predicted_over_observed'] for p in curve])
                error=np.array([np.nan if p is None else p['ratio_uncertainty'] for p in curve])
                valid=np.array([p is not None and p['statistically_valid'] for p in curve])
                ax.plot(PROBABILITIES,np.where(valid,y,np.nan),color=color,label=role)
                ax.fill_between(PROBABILITIES,np.where(valid,y-error,np.nan),np.where(valid,y+error,np.nan),color=color,alpha=.15)
                ax.scatter(PROBABILITIES[~valid],y[~valid],s=5,c=color,alpha=.25,marker='x')
                candidate=entry['calibration_candidate'] if role=='calibration' else entry['checking_point']
                if candidate is not None:
                    x=entry['calibration_candidate']['AE_percentile_within_preselection']
                    ax.plot(x,candidate['predicted_over_observed'],'*',c=color,ms=13)
            ax.axhline(1,c='black',ls='--')
            ax.set(xlabel='AE cut percentile inside selected QCD',ylabel='Predicted / observed QCD',
                   title=f'{branch}, {f:.0%}: frozen {plan["primary_overall_QCD_A_target"]:.1%} overall A target',ylim=(0,2))
            ax.legend(fontsize=8)
    finish(fig,output/'matched_acceptance_cut_curves.png')

    fig_qcd,axes_qcd=plt.subplots(n_branches,n_fractions,figsize=(6*n_fractions,5*n_branches),squeeze=False)
    fig_regions,axes_regions=plt.subplots(n_branches,n_fractions,figsize=(6*n_fractions,5*n_branches),squeeze=False)
    skipped={};display_limits={}
    for row,branch in enumerate(BRANCHES):
        for col,f in enumerate(FRACTIONS):
            key=fraction_name(f);selected=subset(checking[branch],cuts[key]);qc=qcd(selected)
            candidate=results[branch][key][primary]['checking_point']
            pointcuts=None if candidate is None else (candidate['ae_cut'],candidate['md_cut'])
            limits=[visible_range(qc[k],qc['weight'],() if pointcuts is None else [pointcuts[i]]) for i,k in enumerate(('ae','md'))]
            ax=axes_qcd[row,col];mesh=density_hist(ax,qc,limits)
            if mesh is not None:fig_qcd.colorbar(mesh,ax=ax,label='Physical QCD yield')
            score_axes(ax,pointcuts,limits,f'{branch}, {f:.0%}: '+('no candidate' if pointcuts is None else 'fixed primary cuts'))
            ax=axes_regions[row,col]
            if candidate is not None:
                mass=sum(candidate['regions'][r]['yield'] for r in 'ABCD')
                ax.bar(range(4),[candidate['regions'][r]['yield']/mass for r in 'ABCD'],
                       yerr=[np.sqrt(candidate['regions'][r]['sumw2'])/mass for r in 'ABCD'])
                ax.set_xticks(range(4),[f'{r}\nESS={candidate["regions"][r]["effective_count"]:.0f}' for r in 'ABCD'])
            else:ax.text(.5,.5,'No calibration candidate',ha='center',transform=ax.transAxes)
            ax.set(ylabel='Fraction of selected QCD yield',title=f'{branch}, {f:.0%}')
            # Separate standard-size KDEs for each model/retention, no double weighting.
            class_limits=[]
            for label in names:
                mask=selected['label']==label
                if np.any(mask & (selected['weight']>0)):
                    class_limits.append([visible_range(selected[k][mask],selected['weight'][mask],
                                                       () if pointcuts is None else [pointcuts[i]]) for i,k in enumerate(('ae','md'))])
            limits=[(min(x[i][0] for x in class_limits),max(x[i][1] for x in class_limits)) for i in range(2)]
            gx,gy=np.meshgrid(np.linspace(*np.log10(limits[0]),kde_grid),np.linspace(*np.log10(limits[1]),kde_grid))
            fig,ax=plt.subplots(figsize=(8,6));handles=[];missing=[]
            for i,(label,name) in enumerate(names.items()):
                mask=selected['label']==label
                print(f'KDE {branch}, {f:.0%}, {name}: {int(mask.sum()):,} rows',flush=True)
                density=weighted_kde(selected['ae'][mask],selected['md'][mask],selected['weight'][mask],gx,gy)
                if density is None or not np.isfinite(density).all() or density.max()<=0:missing.append(name);continue
                color=PALETTE[i%len(PALETTE)]
                ax.contour(10**gx,10**gy,density,levels=density.max()*np.array([.05,.15,.3,.5,.7,.88]),colors=color,alpha=.7)
                handles.append(Line2D([0],[0],color=color,label=name))
            score_axes(ax,pointcuts,limits,f'{branch}: {f:.0%} retained; weighted class KDE shapes')
            if handles:ax.legend(handles=handles,fontsize=9)
            folder=output/branch/key;folder.mkdir(parents=True)
            finish(fig,folder/'hist2d_pca_md_kde.png')
            skipped[f'{branch}/{key}']=missing;display_limits[f'{branch}/{key}']=limits
    finish(fig_qcd,output/'QCD_score_planes.png')
    finish(fig_regions,output/'ABCD_region_yields.png')
    return {'KDE_skipped_classes':skipped,'class_display_limits':display_limits,
            'KDE_weights':'original physics weights once; per-class normalized shapes, not relative rates',
            'display_quantiles':[.001,.999],
            'plots_never_rank_checking_cuts_or_retention_fractions':True,
            'signal_sample_evaluated':False}
