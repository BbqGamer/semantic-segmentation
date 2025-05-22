import argparse
from pathlib import Path
import random
import tempfile
import shutil
import time

SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"] 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--outfile")
    parser.add_argument("--fraction", type=float, default=0.02)

    args = parser.parse_args()
    dataset = Path(args.dataset)
    outfile = Path(args.outfile).stem

    with tempfile.TemporaryDirectory() as directory:
        for seq in SEQUENCES:
            seqpath = "dataset/sequences/" + seq
            velodyne_dir = dataset / seqpath / "velodyne"
            label_dir = dataset / seqpath / "labels"
            frame_files = list(velodyne_dir.iterdir())
            frame_sample = random.sample(frame_files, int(len(frame_files) * args.fraction))
            
            res_velodyne = Path(directory) / seqpath / "velodyne"
            res_labels = Path(directory) / seqpath / "labels"

            res_velodyne.mkdir(parents=True)
            res_labels.mkdir(parents=True)
            for file in frame_sample:
                shutil.copyfile(file, res_velodyne / file.name)
                lab_file = file.stem + ".label"
                shutil.copyfile(label_dir / lab_file, res_labels / lab_file)

        shutil.make_archive(outfile, "zip", directory)

