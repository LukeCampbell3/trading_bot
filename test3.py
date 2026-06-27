import zipfile, os, tempfile
import h5py

keras_path = r"C:/Users/jcthi/Code/HFT/model/instinct_model.keras"
# C:\Users\jcthi\Code\HFT\model\instinct_model.keras

with zipfile.ZipFile(keras_path, "r") as z:
    assert "model.weights.h5" in z.namelist(), "No model.weights.h5 in archive"
    info = z.getinfo("model.weights.h5")
    print("model.weights.h5 compressed size:", info.compress_size)
    print("model.weights.h5 file size:", info.file_size)

    # extract to temp and inspect HDF5 contents
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "model.weights.h5")
        with z.open("model.weights.h5") as src, open(out, "wb") as dst:
            dst.write(src.read())

        print("extracted weights size:", os.path.getsize(out))

        with h5py.File(out, "r") as f:
            def walk(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print("DATASET:", name, obj.shape)
            f.visititems(walk)