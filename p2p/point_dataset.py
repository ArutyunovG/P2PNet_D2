from detectron2.data.catalog import DatasetCatalog

import imagesize

def _load_point_dataset(list_file):

    dicts = []
    with open(list_file) as f:

        image_counter = 0
        for line in f.readlines():

            img_path, anno_path = (p.strip() for p in line.split(' '))
            width, height = imagesize.get(img_path)
            record = {
                "file_name": img_path,
                "image_id": image_counter,
                "height": height,
                "width": width,
            }

            image_counter += 1

            instances = []

            with open(anno_path) as anno_f:
                anno_f_lines = (l for l in anno_f.readlines() if l.strip())
                points = []
                for anno_f_line in anno_f_lines:
                    x, y = [float(t.strip()) for t in anno_f_line.split(' ')]
                    points.append((x, y))
 
            instances.extend(
                {"category_id": 1, "point": (x, y)}
                for (x, y) in points
            )

            record["annotations"] = instances
            dicts.append(record)

    return dicts


def register_point_dataset(name, list_file):
    DatasetCatalog.register(name, lambda: _load_point_dataset(list_file))
