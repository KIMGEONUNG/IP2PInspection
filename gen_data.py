from utils import Degrade
from glob import glob
from os.path import expanduser, join
from tqdm import tqdm
from PIL import Image

if __name__ == "__main__":

    path_home = expanduser('~')
    path_dataset = 'dataset-local/imagenet/ILSVRC/Data/CLS-LOC/valid'
    path = join(path_home, path_dataset, "*")
    paths = glob(path)

    degrader = Degrade()

    for i, p in enumerate(tqdm(paths)):
        x = Image.open(p)
        y, t = degrader.random_single_deg(x)
        y = y.resize((256, 256), Image.Resampling.LANCZOS)
        t = t.replace(' ', '_')
        y.save('tmp_c/%05d_%s.jpg' % (i, t))
        if i > 100:
            break

