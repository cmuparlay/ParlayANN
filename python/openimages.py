import csv
from collections import defaultdict

bboxes_path = '/ssd2/mazin/oidv6-train-annotations-bbox.csv'

# Dictionary of images mapped to their bounding box types (labels belonging to each image)
image2bboxes = defaultdict(set)

# Dictionary of bounding boxes mapped to the images labeled with them (images each label belongs to)
bboxes2images = defaultdict(set)

# Eventually want to extract and store entire dataset, for now just take first 1000 rows
num_rows = 1000

with open(bboxes_path, 'r') as csv_file:
  csv_reader = csv.DictReader(csv_file)
  cnt = 0
  for row in csv_reader:
    cnt += 1

    image_id = row['ImageID']
    label_name = row['LabelName']

    image2bboxes[image_id].add(label_name)
    bboxes2images[label_name].add(image_id)

    print(image_id, label_name, row['YMin'])

    if cnt == num_rows: 
      break

for k,v in image2bboxes.items():
  print(k, v)

for k,v in bboxes2images.items():
  print(k,v)

  