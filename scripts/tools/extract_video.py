"""Достаёт видео роллаута из parquet REALM: сохраняет mp4 и несколько опорных кадров."""
import os
import sys

import pandas as pd

parquet_path = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

df = pd.read_parquet(parquet_path)
print("строк в parquet:", len(df), "колонки:", list(df.columns))

for i, row in df.iterrows():
    stem = f"{row['task']}_{row['perturbation']}_rep{row['repeat']}"
    mp4 = os.path.join(out_dir, stem + ".mp4")
    with open(mp4, "wb") as f:
        f.write(row["video"])
    print("записан", mp4, os.path.getsize(mp4), "байт")

    # Опорные кадры: равномерно по эпизоду, чтобы видеть, что делает рука.
    try:
        import av

        container = av.open(mp4)
        stream = container.streams.video[0]
        total = stream.frames
        print("кадров в потоке:", total, "размер:", stream.width, "x", stream.height)
        wanted = [int(total * f) for f in (0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.99)] if total else []
        got = 0
        for n, frame in enumerate(container.decode(video=0)):
            if n in wanted:
                png = os.path.join(out_dir, f"{stem}_f{n:04d}.png")
                frame.to_image().save(png)
                got += 1
        print("сохранено кадров:", got)
    except Exception as exc:  # noqa: BLE001
        print("кадры вытащить не удалось:", exc)
