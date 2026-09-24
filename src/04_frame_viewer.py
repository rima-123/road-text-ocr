# ===== CELL 7: Frame viewer + video selection helpers =====
def read_frame_record(summary, frame_index):
    if not 0 <= frame_index < summary["processed_frames"]:
        raise IndexError("Frame index is out of range.")
    with open(summary["frame_index"], "rb") as idx:
        idx.seek(frame_index * 8)
        offset = struct.unpack("<Q", idx.read(8))[0]
    with open(summary["frame_records"], "rb") as records:
        records.seek(offset)
        return json.loads(records.readline())


def read_exact_frame(video_path, index):
    """Request a decoded frame index; use sequential decoding if seeking fails."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open frame-viewer video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok and abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) < .5:
            return frame
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        for _ in range(index + 1):
            ok, frame = cap.read()
            if not ok:
                raise IndexError(f"Cannot decode frame {index}.")
        return frame
    finally:
        cap.release()


def notebook_preview():
    import ipywidgets as widgets
    from IPython.display import display
    image_widget = widgets.Image(format="jpeg", layout=widgets.Layout(max_width="900px"))
    status = widgets.HTML()
    display(widgets.VBox([status, image_widget]))
    def update(frame, record, total):
        scaled = cv2.resize(frame, (min(900, frame.shape[1]), round(frame.shape[0] * min(1, 900 / frame.shape[1]))))
        ok, jpeg = cv2.imencode(".jpg", scaled)
        if ok:
            image_widget.value = jpeg.tobytes()
        status.value = f"Processing frame {record['frame_number']} / {total or '?'} — {record['timestamp_sec']:.3f} s"
    return update


def frame_viewer(summary):
    """Slider, Previous/Next and Play; zero-based index, one-based visible number."""
    import ipywidgets as widgets
    from IPython.display import display
    total = summary["processed_frames"]
    slider = widgets.IntSlider(value=0, min=0, max=total - 1, description="Frame index", continuous_update=False,
                               layout=widgets.Layout(width="95%"))
    play = widgets.Play(value=0, min=0, max=total - 1, interval=250, description="Play")
    link = widgets.jslink((play, "value"), (slider, "value"))
    previous, following = widgets.Button(description="Previous"), widgets.Button(description="Next")
    image_widget = widgets.Image(format="jpeg", layout=widgets.Layout(max_width="1000px"))
    detail = widgets.HTML()
    def show(change=None):
        i = slider.value
        record = read_frame_record(summary, i)
        frame = read_exact_frame(summary["output_video"], i)
        if frame.shape[1] > 1000:
            frame = cv2.resize(frame, (1000, round(frame.shape[0] * 1000 / frame.shape[1])))
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Cannot render the selected frame.")
        image_widget.value = jpeg.tobytes()
        rows = "".join(f"<tr><td>{html.escape(d['text']) or '(unreadable)'}</td><td>{d['score']:.3f}</td>"
                       f"<td>{html.escape(d['status'])}</td><td>{html.escape(str(d['box']))}</td></tr>"
                       for d in record["detections"])
        detail.value = (f"<p><b>Frame {i + 1} of {total}</b> · source time {record['timestamp_sec']:.3f} s</p>"
                        + ("<table><tr><th>Raw text</th><th>Score</th><th>Status</th><th>Box</th></tr>" + rows + "</table>"
                           if rows else "<p>No text regions detected in this frame.</p>"))
        previous.disabled, following.disabled = i == 0, i == total - 1
    previous.on_click(lambda _: setattr(slider, "value", max(0, slider.value - 1)))
    following.on_click(lambda _: setattr(slider, "value", min(total - 1, slider.value + 1)))
    slider.observe(show, names="value")
    ui = widgets.VBox([widgets.HBox([previous, following, play]), slider, image_widget, detail])
    ui._frame_link = link
    show()
    display(ui)
    return ui


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp")


def find_input_videos(root="/kaggle/input"):
    """Videos attached on Kaggle with 'Add Input' (read-only folder)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def make_demo_video(path, font_dir=None, seconds=8, fps=25, size=(1280, 720), seed=0):
    """Moving synthetic text boards. Only a smoke test for sections 3-8 when no real video exists yet."""
    rng = np.random.default_rng(seed)
    fonts = find_fonts(font_dir, True)
    W, H = size
    boards = []
    for k in range(4):
        try:
            board = render_text_image(random_text(rng, OCRConfig()), rng, fonts, augment=False)
        except RuntimeError:
            continue
        scale = min(3.0, W * 0.6 / board.shape[1], H * 0.18 / board.shape[0])
        board = cv2.resize(board, (max(2, int(board.shape[1] * scale)), max(2, int(board.shape[0] * scale))),
                           interpolation=cv2.INTER_CUBIC)
        boards.append((board, float(rng.uniform(0, W)), int(40 + k * (H - 80) / 4),
                       float(rng.uniform(2, 6)) * (1 if k % 2 else -1)))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write the demo video.")
    sky = np.linspace(200, 120, H, dtype=np.float32)[:, None, None] * np.array([1.0, 0.95, 0.85], np.float32)
    background = np.broadcast_to(sky, (H, W, 3)).astype(np.uint8)
    try:
        for f in range(int(seconds * fps)):
            frame = background.copy()
            for x in range(-(f * 12) % 160 - 160, W, 160):
                cv2.rectangle(frame, (x, H - 60), (x + 80, H - 50), (240, 240, 240), -1)
            for board, x0, y, vx in boards:
                bh, bw = board.shape[:2]
                x = int((x0 + vx * f) % (W + bw)) - bw
                a, b = max(0, x), min(W, x + bw)
                if b > a and y + bh <= H:
                    frame[y:y + bh, a:b] = board[:, a - x:b - x]
            writer.write(frame)
    finally:
        writer.release()
    return str(Path(path).resolve())


def _running_on_kaggle():
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").is_dir()


def choose_video(video_path="", demo_if_missing=False, font_dir=None, allow_upload=True):
    """Explicit path > Kaggle input videos > Colab upload button > (optional) synthetic demo clip."""
    if video_path:
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Kaggle: copy the exact path from the Input panel.")
        return str(path)
    found = find_input_videos()
    if found:
        if len(found) > 1:
            print("Several input videos; using the first (set VIDEO_PATH to choose):", *found[:10], sep="\n  ")
        return found[0]
    files = None
    if allow_upload and not _running_on_kaggle():  # Colab's upload widget only works inside Colab
        try:
            from google.colab import files
        except ImportError:
            files = None
    if files is not None:
        print("Click 'Choose Files' below and pick ONE video; wait until it shows 100% done.\n"
              "If you only see 'Upload widget is only available when the cell has been executed in the current "
              "browser session', run this cell again. Big files: upload via the Files sidebar and set VIDEO_PATH.")
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("Upload exactly one video. Then rerun this cell, or set VIDEO_PATH explicitly.")
        return str(Path(next(iter(uploaded))).resolve())
    if demo_if_missing:
        demo = make_demo_video("demo_text_video.mp4", font_dir=font_dir)
        print("No input video found -> synthetic demo clip (pipeline smoke test only, not a benchmark):", demo)
        return demo
    raise ValueError("No video. On Kaggle attach one with 'Add Input' (it appears under /kaggle/input), "
                     "or set VIDEO_PATH to a file.")
