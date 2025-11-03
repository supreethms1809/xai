from datasets import load_dataset
import pandas as pd

ds = load_dataset("ssuresh/idc-patches")

print(ds)
# Extract only image_id values from the dataset without converting entire dataset to pandas
image_list = [example['image_id'] for example in ds['train']]
print(len(image_list))
def extract_patient_id(s):
    # split into a list
    a = s.split('_')
    # the id is the first index in the list
    patient_id = a[0]
    return patient_id

def extract_target(s):
    # split into a list
    a = s.split('_')
    # the target is part of the string in index 4
    b = a[4]
    # the ytarget i.e. 1 or 2 is the 5th index of the string --> class1
    target = b[5]
    return int(target)

def extract_x(s):
    # split into a list
    a = s.split('_')
    # the x-coodinate is part of the string in index 2
    b = a[2]
    # the x-coordinate starts from the 1st index of ths string --> x1251
    x = b[1:]
    return int(x)

def extract_y(s):
    # split into a list
    a = s.split('_')
    # the y-coodinate is part of the string in index 3
    b = a[3]
    # the y-coordinate starts from the 1st index of ths string --> y351
    y = b[1:]
    return int(y)

df_data = pd.DataFrame(image_list, columns=['image_id'])
# create a new column called 'patient_id'
df_data['patient_id'] = df_data['image_id'].apply(extract_patient_id)
# create a new column called 'x'
df_data['x'] = df_data['image_id'].apply(extract_x)
# create a new column called 'y'
df_data['y'] = df_data['image_id'].apply(extract_y)
# create a new column called 'target'
df_data['target'] = df_data['image_id'].apply(extract_target)

df_data.head(10)