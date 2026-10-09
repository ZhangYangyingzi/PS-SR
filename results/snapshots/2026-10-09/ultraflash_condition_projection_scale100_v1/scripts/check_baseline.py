from collections import Counter
from common import *

def main():
    history=list(csv.DictReader((ROOT/'metrics/val_history.csv').open()))
    current=next(r for r in history if int(r['step'])==0)
    previous=next(r for r in csv.DictReader((PREVIOUS/'metrics/val_history.csv').open()) if int(r['step'])==0)
    differences={k:float(current[k])-float(previous[k]) for k in METRICS}
    assert all(abs(x)<1e-5 for x in differences.values())
    counts={}
    for split,expected in [('train',24),('val',6),('test',6)]:
        data=rows(split);videos=Counter(r['video_id'] for r in data)
        assert len(videos)==expected and set(videos.values())=={4}
        counts[split]={'videos':len(videos),'clips':len(data),'only_manifest_read':True}
    clips=list(csv.DictReader((ROOT/'metrics/val_per_clip_step0000.csv').open()))
    assert len(clips)==24
    write_json(AUDIT/'baseline_reproduction.json',{'passed':True,'step0':current,
        'previous_step8000_baseline':previous,'metric_differences':differences,
        'macro':'per-video four clips mean, then six videos mean',
        'dataset_counts':counts,'test_evaluated':False})
    print('step0 baseline reproduction: PASS',differences)

if __name__=='__main__':main()
