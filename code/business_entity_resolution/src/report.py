"""Render the report only from completed and verified local artifacts."""
import argparse
import datetime
import json
from collections import Counter
from core import *
from prepare import ROOT,CACHE,save
from experiment import load

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])+'\n'
def fmt(v):return 'n/a' if v is None else f'{v:.6f}'
def cell(v):return str(v).replace('|','\\|').replace('\n',' ')

def report(n):
    data,out=load(n)
    def read(name):return json.loads((out/name).read_text(encoding='utf-8'))
    verify=read('verification.json');cr=read('candidate_report_final.json');result=read('final_results.json');models=read('models_complete.json');ci=read('confidence_intervals.json')
    selected=result['selected_model'];best=result['models'][selected];overall=best['overall'];budgets=cr['validation'];cand=budgets['100']
    seed=json.loads((CACHE/f'split_{n}.json').read_text(encoding='utf-8'));indexes={s:json.loads((CACHE/f's{s}.json').read_text(encoding='utf-8')) for s in [2,3]}
    env=json.loads((CACHE/'environment.json').read_text(encoding='utf-8'));sampling=read('training_sampling.json');times=read('retrieval_walltime.json')
    disk=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
    peaks=[read(p.name)['memory'].get('peak_rss_bytes',0) for p in out.glob('retrieval_complete_*.json')]
    conservative_retrieval=sum(peaks)+250*1024**2
    foldcounts=Counter(r['fold'] for r in data)
    error_examples=read(f'{selected}_error_examples.json')
    calibration=read(f'{selected}_thresholds.json')['source_calibration']
    error_rows=[]
    for kind in ['false_positive','false_negative']:
        for e in [x for x in error_examples if x['kind']==kind][:3]:
            error_rows.append([kind,e['source1'],e['target'],cell(e['query']['business_name']),cell(e['candidate']['business_name']),
                               cell(e['query']['business_address']),cell(e['candidate']['business_address'] or '<missing>'),', '.join(e['categories'])])
    text=['# Baseline validation report — Version 1','',
          f'Completed local validation using **{n:,} sampled S1 entities** and **10,320,219 training S2/S3 records**. The tuning-selected model is **{selected}**, with untouched-holdout macro F0.5 **{overall["macro_f0_5"]:.6f}** at threshold **{models["models"][selected]["selected_threshold"]:.6f}**.',
          f'Final candidate recall is **{cand["recall"]["all"]["recall"]:.4%}**. '+('This is below the 99.5% working target: the baseline is verified but retrieval-limited and is not a claim of submission readiness.' if cand['recall']['all']['recall']<.995 else 'The working candidate-recall target was reached on this sample.'),
          'No full test inference, leaderboard output, upload, external data, identity lookup, package download, or pretrained-model download occurred.','',
          '## 1. Context, authority and discrepancies','',
          'Read all of `amazon_ml_challenge_context.md`, `dataset_context.md`, and `codex_operating_instructions.md` before material work. Inspected the official problem statement in `README.md`, `Documentation_template.md`, `dataset_analysis_report.md`, `dataset_summary.json`, `utils/validate_submission.py`, and existing `analyze_dataset.py`. No previous matcher, model, experiment log, environment lock, or validation report was present. The separately referenced `Pasted text(4).txt` and organizer clarifications were not found in the project/parent tree. The official README is available; missing clarifications were not invented.',
          'The context overstates default validator coverage: the actual helper requires `--check-ids`, and its prediction/candidate-subset check only warns. It also contradicts the README about nonexistent-ID rejection. This baseline enforces valid IDs and candidate membership independently and runs the stricter checks. The explicit official beta=.5 formula overrides the imprecise prose about twice the precision weight. Some analysis JSON token examples contain mojibake; raw TSVs were decoded as UTF-8.','',
          '## 2. Split, metric and leakage controls','',
          f'Seed **2026**; S1 entities selected by the lowest seeded BLAKE2 hashes in a full S1 scan. Group union uses suffix-normalized names OR reordered address token keys. There are {seed["group_count"]:,} groups, largest size {seed["largest_group"]}. Fit: {foldcounts["fit"]:,}; tuning: {foldcounts["tune"]:,}; final holdout: {foldcounts["validation"]:,}. Country and 0/1/2/3/4/5+ cardinality strata are present in every fold. IDs, group assignments, and stratum counts are persisted.',
          'Candidate indexes and sampled IDF use only unlabeled target text. Full target records may be shared as retrieval distractors across folds, as they would at inference; no held-out labels enter model fitting, negative sampling, calibration, or threshold choice. Zero shared positive target IDs across folds and zero shared grouping keys were verified. Grouping only covers sampled entities and does not prove absence of looser semantic duplicates.',
          'Metric tests ran before modeling. ID lists are parsed as sets with duplicate rejection. Per-entity F0.5 is `1.25*TP/(0.25*true_count+predicted_count)`, with true-empty/pred-empty=1 and true-empty/pred-nonempty=0. For supplementary macro precision/recall, both-empty is defined as 1, any other zero denominator as 0; the official statement only explicitly specifies singleton F-score. Main metrics are macro over all held-out S1, not pairwise F1.',
          f'The selected model has a 1,000-replicate cluster-bootstrap 95% F0.5 interval of {ci[selected]["cluster_bootstrap_95_percent"]}. This reflects sampling uncertainty, not France/generalization uncertainty. Small difficult slices are less stable.','',
          '## 3. Candidate routes and budgets','',
          'S2 and S3 indexes are separate. Union exact folded names, trailing legal-suffix keys, rare name-token FTS, name character TF-IDF reranking of trigram-FTS seeds, address-token retrieval, reordered exact address keys, numeric/address components, and address-character reranking. All routes retain country-agnostic access; country is an agreement feature, never an accepted-country list.',
          'The fixed final budget is 100 per route per source, with 400 FTS seeds per route; collision-key blocks are ranked before truncation. Numeric retrieval allows up to eight numeric/postal keys and all alphabetic address words. Binary-TF 3–5-gram cosine uses label-free IDF from a fixed-stride ~20k target sample. Character retrieval is approximate seeded top-k, not exact global sparse TF-IDF; address character retrieval depends on lexical seeds. No dense Cartesian similarities are built. No local transliteration dependency was available.',
          'A preliminary fitting-only S2 audit retrieved 168/172 links; four misses had cross-script names and shortened addresses. Broadening numeric/address words recovered three of those four. The interrupted initial run and its code are preserved. This was the only retrieval-policy adjustment, made without final-validation labels.','',
          '## 4. Candidate recall and difficult slices','',
          table(['Slice','True links','Retrieved','Recall'],[[s,v['true_links'],v['retrieved'],fmt(v['recall'])] for s,v in cand['recall'].items()]),
          'Low name similarity means folded 3–5-gram Jaccard <0.2. Cross-script means a non-Latin candidate name whose detected letter-script set differs from the query. Script detection uses Unicode character names; these are operational diagnostics, not language identification. Common-name means at least five exact folded-name records in that source.','',
          '## 5. Candidate growth and reduction ratio','',
          table(['Route budget','Recall','Total pairs','Mean','p50','p90','p99','Max','Reduction ratio'],[[k,fmt(v['recall']['all']['recall']),v['counts']['pairs'],f'{v["counts"]["mean"]:.2f}',v['counts']['p50'],v['counts']['p90'],v['counts']['p99'],v['counts']['max'],fmt(v['counts']['reduction_ratio'])] for k,v in budgets.items()]),
          f'Source pair totals at budget 100: {cand["source_pairs"]}. Reduction denominator is validation S1 count × all 10,320,219 training targets. All true links remain in recall denominators, including unretrieved links. Budget experiments reuse ranked routes and do not inject positives.','',
          f'Oracle macro F0.5 with a perfect classifier restricted to the final candidates: **{cand["oracle_entity_metrics"]["macro_f0_5"]:.6f}**. This is a diagnostic upper bound, not model performance.','',
          '## 6–7. Logistic and boosted-tree comparison','',
          table(['Model','Tune F0.5','Selected threshold','Validation F0.5','Macro precision','Macro recall','Parameters'],[[m,fmt(models['models'][m]['tune']['macro_f0_5']),fmt(models['models'][m]['selected_threshold']),fmt(v['overall']['macro_f0_5']),fmt(v['overall']['macro_precision']),fmt(v['overall']['macro_recall']),models['models'][m]['parameters']] for m,v in result['models'].items()]),
          'No sklearn, SciPy, LightGBM, XGBoost, CatBoost, or rapidfuzz was installed. Both models are original MIT-licensed NumPy implementations trained from scratch: standardized L2 logistic with Adam; histogram Newton gradient-boosted trees with 70 trees, depth 3, 32 bins, learning rate .12 and leaf L2=5. These are practical fallbacks, not claims of parity with mature library implementations. Model JSON includes ordered features, learned parameters, license, and parameter count. Reload predictions match saved scores.',
          f'The same retrieved fitting pairs and weights were used for both models. Negative sampling: {sampling}. No positives absent from retrieval were inserted. All 59 features and their order are saved; they include separate name/address similarities, collision counts, numeric/postal conflicts, scripts, missingness, source, open-country equality, interactions, density, route hits/ranks/scores. Raw fields and versioned views are cached.','',
          '## 8. Threshold selection','',
          'For each fixed model, a broad threshold grid from .001 through 1.0 was followed by 51 refinements around the best tuning threshold. Every candidate at or above threshold is predicted; no top-1 or mandatory-link rule is used. Model selection and thresholds were frozen before final holdout evaluation. No probability calibration is applied. Source reliability bins are saved in each threshold artifact; separate source thresholds were not introduced. France has no special threshold.',
          f'All serious threshold results, including F0.5, macro precision/recall, singleton accuracy, zero-match rate, mean predicted links, distributions, false-positive and false-negative links, are in `{out.relative_to(ROOT)}/{{logistic,boosted}}_thresholds.json`.','',
          'Selected-model tuning reliability bins (uncalibrated scores; source comparison):',
          table(['Source','Score interval','Pairs','Mean score','Positive fraction'],[[s,f'{b["lower"]:.1f}–{b["upper"]:.1f}',b['n'],fmt(b['mean_score']),fmt(b['positive_rate'])] for s,bins in calibration.items() for b in bins]),'',
          '## 9. Selected-model quality by slice','',
          table(['Measure','Value'],[[k,v] for k,v in overall.items() if not isinstance(v,dict)]),
          table(['Entity slice','N','F0.5','Macro precision','Macro recall','Singleton accuracy'],[[s,v['n_entities'],fmt(v['macro_f0_5']),fmt(v['macro_precision']),fmt(v['macro_recall']),fmt(v['singleton_accuracy'])] for s,v in best['entity_slices'].items() if s.startswith(('country_','source_'))]),
          'Source-level entity metrics apply the same singleton convention after restricting each S1 set to that source. The following difficulty metrics are link precision/recall within the slice, not alternative official scores:',
          table(['Link slice','True links','Predicted','TP','Precision','Recall'],[[s,v.get('truth',0),v.get('predicted',0),v.get('tp',0),fmt(v['precision']),fmt(v['recall'])] for s,v in best['link_slices'].items()]),
          'Unlabeled sanity checks processed the first 1,000 records in each test source, preserving French and other Unicode text and open country strings. No test entity was retrieved or scored. Details are in `unlabeled_schema_sanity.json`; this proves schema handling, not French matching quality.','',
          '## 10. Match-cardinality diagnostics','',
          table(['Slice (5 = 5+)','N','F0.5','Precision','Recall','Mean predicted links'],[[s,v['n_entities'],fmt(v['macro_f0_5']),fmt(v['macro_precision']),fmt(v['macro_recall']),f'{v["average_links"]:.3f}'] for s,v in best['entity_slices'].items() if 'cardinality_' in s]),
          f'True distribution: {overall["true_counts"]}. Prediction distribution: {overall["prediction_counts"]}.','',
          '## 11. False merges and missed links','',
          table(['Error type','Category','Links'],[[kind,c,count] for kind,counts in best['error_categories'].items() for c,count in sorted(counts.items(),key=lambda x:-x[1])]),
          'Categories overlap. Missing addresses remove disambiguating evidence from identical/common names; strong address similarity may connect unrelated co-located entities. Cross-script names, truncated addresses and changed numbers can defeat all literal retrieval routes. Retrieved false negatives reflect the precision-heavy threshold. Full error IDs/categories and retrieval-miss text examples are saved in model error and retrieval-miss JSON files. Do not infer semantic label errors from these diagnostics.','',
          'Examples below show the three highest-scoring false positives and the first three false negatives in deterministic S1/ID order. They are inspection examples, not a representative error sample:',
          table(['Error','S1','Target','Query name','Candidate name','Query address','Candidate address','Categories'],error_rows),'',
          '## 12. Runtime, RAM, disk and caches','',
          table(['Stage','Seconds','Peak process RAM MiB','Disk MiB'],[[f'S{src} full index',f'{v["seconds"]:.2f}',f'{v["memory"].get("peak_rss_bytes",0)/2**20:.1f}',f'{v["bytes"]/2**20:.1f}'] for src,v in indexes.items()]+[['Final retrieval wall time',f'{times["seconds"]:.2f}',f'{max(peaks)/2**20:.1f} per worker','see artifacts'],['Fit/score/threshold stages',f'{models["seconds"]:.2f}',f'{models["memory"].get("peak_rss_bytes",0)/2**20:.1f}','see models'],['Verification',f'{verify["seconds"]:.2f}',f'{verify["memory"].get("peak_rss_bytes",0)/2**20:.1f}','see fixture']]),
          f'Total `artifacts/v1` disk at reporting: {disk/2**30:.3f} GiB, including the retained preliminary run and verification fixture. Conservative concurrent retrieval RAM bound (sum of worker peaks plus 250 MiB parent allowance): {conservative_retrieval/2**30:.3f} GiB; worker peaks need not occur simultaneously. Windows process counters are measured; smoke counters before the API handle fix were unavailable. These timings exclude interactive implementation/review time. No full-data inference runtime is claimed.',
          f'A simple linear extrapolation of observed query throughput to 1,732,544 test S1 rows is {times["seconds"]*1732544/n/86400:.1f} days for retrieval at the same worker count, excluding index building and model scoring. This is an estimate, not a run: the SQLite/Python baseline needs substantial throughput engineering before full-scale use.',
          'Index SQLite files, IDF caches, source/data/code fingerprints, split IDs, per-shard NPZ features, exact candidate audit JSONL, normalized-view JSONL, model JSON, score NPZ, threshold curves, confidence intervals, local validation TSVs, verifier logs, environment, and experiment logs are persisted.','',
          '## 13. Exact reproduction commands','',
          'Run from the project root in PowerShell:',
          '```powershell\n& ./code/business_entity_resolution/reproduce.ps1 -Stage Smoke\n& ./code/business_entity_resolution/reproduce.ps1 -Stage Full -Workers 8\n# Reuse completed retrieval artifacts:\n& ./code/business_entity_resolution/reproduce.ps1 -Stage Evaluate\n```',
          f'Interpreter: `{env["executable"]}`. Python 3.12.14; NumPy 2.3.5; SQLite 3.53.1. Exact stage arguments are in the script and run configuration. On another host change the interpreter path only and provide the pinned environment with SQLite FTS5. Resume only matching-fingerprint caches.','',
          '## 14. Verification and compliance','',
          table(['Check','Result'],[[k,v] for k,v in verify.items() if k not in ['memory','seconds']]),
          'Automated tests cover metric edge cases, official example, TSV parsing, duplicate/unknown/source ID checks, normalization, split grouping, candidate deduplication, multiple/empty predictions, fallback models, and serialization. Both local validation outputs pass the organizer validator with ID checking enabled. Its `test_source*.tsv` interface is supplied with a clearly marked **training validation fixture**, not challenge test predictions. Exact candidate membership is independently asserted. All seven original TSV SHA256 hashes remain unchanged.',
          'Only supplied training data provided resolution evidence. Bounded test reads were unlabeled schema checks. No internet requests, external lookup, competitor code, pretrained weights, installations, uploads, final submission filenames, or final package were used. The newly authored models/code are explicitly MIT licensed and far below 8B parameters. No third-party pretrained model license is assumed.','',
          '## 15. Limitations and next three experiments','',
          'Completed: exact metric tests, full-corpus retrieval for sampled S1, candidate diagnostics, two trained baselines, tuning-only threshold/model selection, untouched holdout scoring, local artifact verification and this report. The 99.5% candidate target is a working goal; unmet recall is explicitly a blocker to calling this a strong final system. Exact global 3–5-gram TF-IDF retrieval and a mature boosted-tree library remain unimplemented alternatives, not completed claims. Final-validation results must not be used to retune this same holdout.',
          '1. Improve cross-script retrieval using only fitting data: learn transliteration/name-variant mappings from official fitting links with grouped cross-fitting, and evaluate numeric/address recall when numbers change. Compare on a fresh development split before a new holdout.',
          '2. Replace lexical-seeded address character retrieval with scalable sparse character retrieval and test larger seed budgets versus candidate growth. Separate no-shared-token misses from rank truncation, retaining the full target corpus.',
          '3. With retrieval fixed, compare a license-compatible mature boosted-tree implementation if made available locally, emphasizing common-name/missing-address precision and country/script holdout robustness. Fit any calibration on a separate fitting fold and retain an untouched final holdout.',
          'Stopped at local baseline validation. Full test inference and any submission activity require a subsequent explicit request.','']
    (ROOT/'baseline_validation_report.md').write_text('\n'.join(text),encoding='utf-8')
    record=dict(experiment_id='v1_seed2026_n6000_retrieval2',timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      git_commit_or_code_fingerprint=read('code_fingerprints.json'),data_fingerprint=json.loads((CACHE/'data_fingerprints.json').read_text(encoding='utf-8')),
      split_id=f'split_{n}',random_seed=SEED,candidate_routes=read('config.json')['routes'],candidate_budgets=read('config.json'),
      candidate_recall_overall=cand['recall']['all']['recall'],candidate_recall_s2=cand['recall']['S2']['recall'],candidate_recall_s3=cand['recall']['S3']['recall'],
      candidate_count_statistics=cand['counts'],model_and_version={m:{'implementation_version':'numpy-original-v1.0','parameters':v['parameters']} for m,v in models['models'].items()},feature_set='59 features in retrieval.py FEATURES',
      negative_sampling=sampling,thresholds={m:v['selected_threshold'] for m,v in models['models'].items()},macro_f0_5=overall['macro_f0_5'],macro_precision=overall['macro_precision'],
      macro_recall=overall['macro_recall'],singleton_metrics={k:v for k,v in overall.items() if 'singleton' in k},slice_metrics=best['link_slices'],
      runtime={'retrieval':times,'models':models['seconds'],'verification':verify['seconds']},peak_memory={'retrieval_upper_bound_bytes':conservative_retrieval,'models':models['memory']},
      artifact_paths=[str(out.relative_to(ROOT)),'baseline_validation_report.md'],notes='Verified local baseline; no test inference. Retrieval target status and fallbacks explicitly documented.')
    with open(ROOT/'experiment_log.jsonl','a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(ROOT/'baseline_validation_report.md',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=6000);a=p.parse_args();report(a.n)
