import os
import sys

from .utils import load_obj, save_obj

utils_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

wwcd = os.path.join(parent_path, 'data', 'intermediate', 'write_with_class')
write_class = load_obj(wwcd)

writers = []  # each entry is a (writer, [list of (file, class)]) tuple
cimages = []
(cw, _, _) = write_class[0]
for (w, f, c) in write_class:
    if w != cw:
        writers.append((cw, cimages))
        cw = w
        cimages = [(f, c)]
    cimages.append((f, c))
writers.append((cw, cimages))

ibwd = os.path.join(parent_path, 'data', 'intermediate', 'images_by_writer')
save_obj(writers, ibwd)
