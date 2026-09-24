# ===== Demo: run on several videos, show video + confident readings =====
from IPython.display import Video, display

DEMO_VIDEOS = [MY_VIDEOS[2], MY_VIDEOS[0]]     # raat, din
MIN_VOTE, MIN_FRAMES = 0.5, 25

SPOTTER_HF.detector.post.update(box_thresh=0.45, bin_thresh=0.3, infer_side=2000)
DEMO_RUNS = []
for _v in DEMO_VIDEOS:
    print("\n" + "=" * 70 + f"\ninput: {_v}")
    display(Video(_v, embed=True, width=800))
    _run = process_road_video(_v, SPOTTER_HF, ROAD_SAVE_ROOT / "my_video_runs", overlay="fused")
    DEMO_RUNS.append(_run)
    with open(_run["tracks_csv"], encoding="utf-8-sig", newline="") as _fh:
        _rows = [r for r in csv.DictReader(_fh) if r.get("text")
                 and int(r["frames"]) >= MIN_FRAMES and float(r["vote_share"]) >= MIN_VOTE]
    _rows.sort(key=lambda r: -int(r["frames"]))
    print(f"\n{len(_rows)} confident readings (vote >= {MIN_VOTE:.0%}, >= {MIN_FRAMES} frames):")
    for _r in _rows:
        print(f"  {_r['text']:<20} vote {float(_r['vote_share']):.0%}  frames {_r['frames']}")
    _mb = Path(_run["output_video"]).stat().st_size / 1024**2
    print(f"\nAnnotated video ({_mb:.0f} MB): {_run['output_video']}")
    if _mb <= 60:
        display(Video(_run["output_video"], embed=True, width=900))

MY_RUN = DEMO_RUNS[-1]
MY_VIEWER = frame_viewer(MY_RUN)
