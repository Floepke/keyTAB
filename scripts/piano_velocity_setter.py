import json


input_file = '/home/flop/pCloudDrive/keyTAB projects/Kaars.piano'
velocity = 80
output_file = '/home/flop/pCloudDrive/keyTAB projects/KaarsV.piano'

with open(input_file, 'r') as f:
    data = json.load(f)

for note in data['events']['note']:
    note['velocity'] = velocity

with open(output_file, 'w') as f:
    json.dump(data, f)
