#Import modules
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier
#Performance Measurement Modules
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
#Modules for encoding categorical features
from sklearn.preprocessing import StandardScaler, LabelEncoder

#pdbbox for interpretation
from pdpbox import pdp, info_plots

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
sns.set_palette("bright")

df = pd.read_csv('./StudentsData.csv')
df.head()
len(df)
#A function to categorize the numerical grades to letter grades
def score_to_grade(score):
    if score >= 80:
        return 'A'
    if score >= 60:
        return 'B'
    if score >= 40:
        return 'C'
    return 'F'

#Convert numerical grades to categorical letter grades and add three new features
df['math grade'] = df['math score'].apply(score_to_grade)
df['reading grade'] = df['reading score'].apply(score_to_grade)
df['writing grade'] = df['writing score'].apply(score_to_grade)
#Display the modified data set
df.head()

df['gender'].value_counts() / len(df) * 100

#A function to create bar charts
def plot_vc_bar_charts(df, col, ax, color='b'):
    df_vc = df[col].value_counts() / len(df) * 100
    df_vc = df_vc.reset_index()
    df_vc = df_vc.rename(columns={'count': 'pc'})
    
    bar_width = 0.35
    opacity = 0.9
    # After reset_index, the original column name is preserved, we just rename the count column to 'pc'
    labels = df_vc[col].values
    capitalized_labels = []
    for label in labels:
        words = label.split(' ')
        capitalized_label = ' '.join([w.capitalize() for w in words])
        capitalized_labels.append(capitalized_label)
    sizes = df_vc['pc'].values
    index = np.arange(len(sizes))
    
    ax.bar(index, sizes, bar_width,
           alpha=opacity,
           label=labels,
           color=color)
    ax.set_xticks(index)
    ax.set_xticklabels(capitalized_labels)
    ax.grid(True)

print(df.columns)
print(df['race/ethnicity'])
#Display the distribution of Ethnicity and Parent level of Education
f, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_vc_bar_charts(df, 'race/ethnicity', ax=ax[0], color='r')
plot_vc_bar_charts(df, 'parental level of education', ax=ax[1], color='g')
ax[0].set_title('Race / Ethnicity')
ax[1].set_title('Parent Level of Education')
ax[0].set_ylabel('% of Population')
ax[1].set_ylabel('% of Population')
ax[0].set_ylim([0, 35])
ax[1].set_ylim([0, 25])

def get_stats(df, grade_col):
    df_grade = df.groupby([grade_col]).agg({'lunch': 'count'}).reset_index()
    df_grade = df_grade.rename(columns={'lunch': 'count'})
    df_grade['pc'] = df_grade['count'] / len(df) * 100
    
    df_gender = df.groupby(['gender', grade_col]).agg({'lunch': 'count'}).reset_index()
    df_gender = df_gender.rename(columns={'lunch': 'count'})
    df_gender['pc'] = df_gender['count'] / len(df) * 100
    
    df_race = df.groupby(['race/ethnicity', grade_col]).agg({'lunch': 'count'}).reset_index()
    df_race = df_race.rename(columns={'lunch': 'count'})
    df_race['pc'] = df_race['count'] / len(df) * 100
    
    df_parent = df.groupby(['parental level of education', grade_col]).agg({'lunch': 'count'}).reset_index()
    df_parent = df_parent.rename(columns={'lunch': 'count'})
    df_parent['pc'] = df_parent['count'] / len(df) * 100
    
    df_lunch = df.groupby(['lunch', grade_col]).agg({'gender': 'count'}).reset_index()
    df_lunch = df_lunch.rename(columns={'gender': 'count'})
    df_lunch['pc'] = df_lunch['count'] / len(df) * 100
    
    df_test_prep = df.groupby(['test preparation course', grade_col]).agg({'gender': 'count'}).reset_index()
    df_test_prep = df_test_prep.rename(columns={'gender': 'count'})
    df_test_prep['pc'] = df_lunch['count'] / len(df) * 100
    
    return df_grade, df_gender, df_race, df_parent, df_lunch, df_test_prep

df_grade_math, df_gender_math, df_race_math, df_parent_math, df_lunch_math, df_test_prep_math = get_stats(df, 'math grade')
df_grade_reading, df_gender_reading, df_race_reading, df_parent_reading, df_lunch_reading, df_test_prep_reading = get_stats(df, 'reading grade')
df_grade_writing, df_gender_writing, df_race_writing, df_parent_writing, df_lunch_writing, df_test_prep_writing = get_stats(df, 'writing grade')

fig, ax = plt.subplots(figsize=(8, 5))
grades = ['A', 'B', 'C', 'F']
index = np.arange(len(grades))
bar_width = 0.25

opacity = 0.8
error_config = {'ecolor': '0.3'}

ax.bar(index, df_grade_math['pc'], bar_width,
       alpha=opacity, color='r',
       label='Math',
       hatch='x')
ax.bar(index + bar_width, df_grade_reading['pc'], bar_width,
       alpha=opacity, color='g',
       label='Reading',
       hatch='o')
ax.bar(index + bar_width * 2, df_grade_writing['pc'], bar_width,
       alpha=opacity, color='b',
       label='Writing',
       hatch='+')

ax.set_xlabel('Grades')
ax.set_ylabel('% of Population')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(grades)
ax.legend()
ax.grid(True)

fig, ax = plt.subplots(3, 1, figsize=(10, 20))

grades = ['A', 'B', 'C', 'F']
index = np.arange(len(grades))
bar_width = 0.35

opacity = 0.9
error_config = {'ecolor': '0.3'}

ax[0].bar(index, df_gender_math[df_gender_math['gender'] == 'male']['pc'], bar_width,
                alpha=opacity, color='b',
                label='Male')
ax[0].bar(index + bar_width, df_gender_math[df_gender_math['gender'] == 'female']['pc'], bar_width,
                alpha=opacity, color='g',
                label='Female')

ax[0].set_xlabel('Grades')
ax[0].set_ylabel('% of Population')
ax[0].set_title('Math Grades by Gender')
ax[0].set_xticks(index + bar_width / 2)
ax[0].set_xticklabels(grades)
ax[0].legend()
ax[0].grid(True)

ax[1].bar(index, df_gender_reading[df_gender_reading['gender'] == 'male']['pc'], bar_width,
                alpha=opacity, color='b',
                label='Male')
ax[1].bar(index + bar_width, df_gender_reading[df_gender_reading['gender'] == 'female']['pc'], bar_width,
                alpha=opacity, color='g',
                label='Female')

ax[1].set_xlabel('Grades')
ax[1].set_ylabel('% of Population')
ax[1].set_title('Reading Grades by Gender')
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(grades)
ax[1].legend()
ax[1].grid(True)

ax[2].bar(index, df_gender_writing[df_gender_writing['gender'] == 'male']['pc'], bar_width,
                alpha=opacity, color='b',
                label='Male')
ax[2].bar(index + bar_width, df_gender_writing[df_gender_writing['gender'] == 'female']['pc'], bar_width,
                alpha=opacity, color='g',
                label='Female')

ax[2].set_xlabel('Grades')
ax[2].set_ylabel('% of Population')
ax[2].set_title('Writing Grades by Gender')
ax[2].set_xticks(index + bar_width / 2)
ax[2].set_xticklabels(grades)
ax[2].legend()
ax[2].grid(True)

fig, ax = plt.subplots(3, 1, figsize=(10, 15))

grades = ['A', 'B', 'C', 'F']
index = np.arange(len(grades))
bar_width = 0.35

opacity = 0.8
error_config = {'ecolor': '0.3'}

math_a_sum = df_race_math[df_race_math['race/ethnicity'] == 'group A']['pc'].sum()
math_c_sum = df_race_math[df_race_math['race/ethnicity'] == 'group C']['pc'].sum()
ax[0].bar(index, df_race_math[df_race_math['race/ethnicity'] == 'group A']['pc'] / math_a_sum * 100, bar_width,
                alpha=opacity, color='r',
                label='Group A')
ax[0].bar(index + bar_width, df_race_math[df_race_math['race/ethnicity'] == 'group C']['pc'] / math_c_sum * 100, bar_width,
                alpha=opacity, color='g',
                label='Group C')

ax[0].set_xlabel('')
ax[0].set_ylabel('% of Population in Group')
ax[0].set_title('Math Grades by Ethnicity')
ax[0].set_xticks(index + bar_width / 2)
ax[0].set_xticklabels(grades)
ax[0].legend()
#     ax[0].set_ylim([0, 16])
ax[0].grid(True)

read_a_sum = df_race_reading[df_race_reading['race/ethnicity'] == 'group A']['pc'].sum()
read_c_sum = df_race_reading[df_race_reading['race/ethnicity'] == 'group C']['pc'].sum()
ax[1].bar(index, df_race_reading[df_race_reading['race/ethnicity'] == 'group A']['pc'] / read_a_sum * 100, bar_width,
                alpha=opacity, color='r',
                label='Group A')
ax[1].bar(index + bar_width, df_race_reading[df_race_reading['race/ethnicity'] == 'group C']['pc'] / read_c_sum * 100, bar_width,
                alpha=opacity, color='g',
                label='Group C')

ax[1].set_xlabel('')
ax[1].set_ylabel('% of Population in Group')
ax[1].set_title('Reading Grades by Ethnicity')
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(grades)
ax[1].legend()
#     ax[1].set_ylim([0, 18])
ax[1].grid(True)

write_a_sum = df_race_writing[df_race_writing['race/ethnicity'] == 'group A']['pc'].sum()
write_c_sum = df_race_writing[df_race_writing['race/ethnicity'] == 'group C']['pc'].sum()
ax[2].bar(index, df_race_writing[df_race_writing['race/ethnicity'] == 'group A']['pc'] / write_a_sum * 100, bar_width,
                alpha=opacity, color='r',
                label='Group A')
ax[2].bar(index + bar_width, df_race_writing[df_race_writing['race/ethnicity'] == 'group C']['pc'] / write_c_sum * 100, bar_width,
                alpha=opacity, color='g',
                label='Group C')

ax[2].set_xlabel('Grades')
ax[2].set_ylabel('% of Population in Group')
ax[2].set_title('Writing Grades by Ethnicity')
ax[2].set_xticks(index + bar_width / 2)
ax[2].set_xticklabels(grades)
ax[2].legend()
#     ax[2].set_ylim([0, 16])
ax[2].grid(True)

fig, ax = plt.subplots(3, 1, figsize=(10, 15))

grades = ['A', 'B', 'C', 'F']
index = np.arange(len(grades))
bar_width = 0.35

opacity = 0.8
error_config = {'ecolor': '0.3'}

math_a_sum = df_parent_math[df_parent_math['parental level of education'] == 'high school']['pc'].sum()
math_c_sum = df_parent_math[df_parent_math['parental level of education'] == 'master\'s degree']['pc'].sum()
math_ma_pc = np.array(df_parent_math[df_parent_math['parental level of education'] == 'master\'s degree']['pc'].values) / math_c_sum * 100
math_ma_pc = np.append(math_ma_pc, 0) # for grade f
ax[0].bar(index, df_parent_math[df_parent_math['parental level of education'] == 'high school']['pc'] / math_a_sum * 100, bar_width,
                alpha=opacity, color='r',
                label='High School')
ax[0].bar(index + bar_width, 
          math_ma_pc, bar_width,
                alpha=opacity, color='g',
                label='Master\'s Degree')

ax[0].set_xlabel('')
ax[0].set_ylabel('% of Population in Group')
ax[0].set_title('Math Grades by Parent Level of Education')
ax[0].set_xticks(index + bar_width / 2)
ax[0].set_xticklabels(grades)
ax[0].legend()
#     ax[0].set_ylim([0, 16])
ax[0].grid(True)

read_a_sum = df_parent_reading[df_parent_reading['parental level of education'] == 'high school']['pc'].sum()
read_c_sum = df_parent_reading[df_parent_reading['parental level of education'] == 'master\'s degree']['pc'].sum()
read_ma_pc = np.array(df_parent_reading[df_parent_reading['parental level of education'] == 'master\'s degree']['pc'].values) / read_c_sum * 100
read_ma_pc = np.append(read_ma_pc, 0) # for grade f
ax[1].bar(index, df_parent_reading[df_parent_reading['parental level of education'] == 'high school']['pc'] / read_a_sum * 100, bar_width,
                alpha=opacity, color='r',
                label='High School')
ax[1].bar(index + bar_width, read_ma_pc, bar_width,
                alpha=opacity, color='g',
                label='Master\'s Degree')

ax[1].set_xlabel('')
ax[1].set_ylabel('% of Population in Group')
ax[1].set_title('Reading Grades by Parent Level of Education')
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(grades)
ax[1].legend()
#     ax[1].set_ylim([0, 18])
ax[1].grid(True)

write_a_sum = df_parent_writing[df_parent_writing['parental level of education'] == 'high school']['pc'].sum()
write_c_sum = df_parent_writing[df_parent_writing['parental level of education'] == 'master\'s degree']['pc'].sum()
write_ma_pc = np.array(df_parent_writing[df_parent_writing['parental level of education'] == 'master\'s degree']['pc'].values) / write_c_sum * 100
write_ma_pc = np.append(write_ma_pc, 0) # for grade f
ax[2].bar(index, df_parent_writing[df_parent_writing['parental level of education'] == 'high school']['pc'] / write_a_sum * 100, bar_width,
                alpha=opacity, color='r',
                label='High School')
ax[2].bar(index + bar_width, write_ma_pc, bar_width,
                alpha=opacity, color='g',
                label='Master\'s Degree')

ax[2].set_xlabel('Grades')
ax[2].set_ylabel('% of Population in Group')
ax[2].set_title('Writing Grades by Parent Level of Education')
ax[2].set_xticks(index + bar_width / 2)
ax[2].set_xticklabels(grades)
ax[2].legend()
#     ax[2].set_ylim([0, 16])
ax[2].grid(True)

ple = ['high school', 'some high school', 'some college', "bachelor's degree", "associate's degree", "master's degree"]
ple_map = {}
for idx, p in enumerate(ple):
    ple_map[p] = idx

# First, encode the input features
gender_le = LabelEncoder()
race_le = LabelEncoder()
parent_le = LabelEncoder()
lunch_le = LabelEncoder()
test_prep_le = LabelEncoder()

gender_le.fit(df['gender'])
race_le.fit(df['race/ethnicity'])
parent_le.fit(df['parental level of education'])
lunch_le.fit(df['lunch'])
test_prep_le.fit(df['test preparation course'])

df['gender_le'] = gender_le.transform(df['gender'])
df['race_le'] = race_le.transform(df['race/ethnicity'])
df['parent_le'] = df['parental level of education'].apply(lambda x: ple_map[x])
df['lunch_le'] = lunch_le.transform(df['lunch'])
df['test_prep_le'] = test_prep_le.transform(df['test preparation course'])

# Next, encode the target variables
math_grade_le = LabelEncoder()
reading_grade_le = LabelEncoder()
writing_grade_le = LabelEncoder()

math_grade_le.fit(df['math grade'])
reading_grade_le.fit(df['reading grade'])
writing_grade_le.fit(df['writing grade'])

df['math_grade_le'] = math_grade_le.transform(df['math grade'])
df['reading_grade_le'] = reading_grade_le.transform(df['reading grade'])
df['writing_grade_le'] = writing_grade_le.transform(df['writing grade'])

df_train_val, df_test = train_test_split(df, test_size=0.2, stratify=df['math_grade_le'], 
                                         shuffle=True, random_state=42)
df_train_val = df_train_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

feature_cols = ['gender_le', 'race_le', 'parent_le', 'lunch_le', 'test_prep_le']

X_train_val = df_train_val[feature_cols]
X_test = df_test[feature_cols]
y_math_train_val = df_train_val['math_grade_le']
y_reading_train_val = df_train_val['reading_grade_le']
y_writing_train_val = df_train_val['writing_grade_le']
y_math_test = df_test['math_grade_le']
y_reading_test = df_test['reading_grade_le']
y_writing_test = df_test['writing_grade_le']

def create_random_forest_model(n_estimators, 
                               max_depth=10,
                               criterion='gini',
                               random_state=42,
                               n_jobs=4):
    return RandomForestClassifier(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  criterion=criterion,
                                  random_state=random_state,
                                  n_jobs=n_jobs)

math_model = create_random_forest_model(50)
math_model.fit(X_train_val, y_math_train_val)

y_math_model_test = math_model.predict(X_test)
y_math_model_proba_test = math_model.predict_proba(X_test)
print(y_math_model_proba_test)

y_math_baseline = [1 for _ in range(len(y_math_model_proba_test))]

acc = math_model.score(X_test, y_math_test)
print(acc)

from sklearn.metrics import recall_score, precision_score, f1_score

precision = precision_score(y_math_test, y_math_model_test, average="weighted")
p_baseline = precision_score(y_math_test, y_math_baseline, average="weighted")
r_baseline = recall_score(y_math_test, y_math_baseline, average="weighted")
recall = recall_score(y_math_test, y_math_model_test, average="weighted")
f1 = f1_score(y_math_test, y_math_model_test, average="weighted")
f1_baseline = f1_score(y_math_test, y_math_baseline, average="weighted")
print(precision)
print(p_baseline)
print(recall)
print(r_baseline)
print(f1)
print(f1_baseline)

reading_model = create_random_forest_model(25)
reading_model.fit(X_train_val, y_reading_train_val)

y_reading_model_test = reading_model.predict(X_test)
y_reading_model_proba_test = reading_model.predict_proba(X_test)

y_reading_baseline = [1 for _ in range(len(y_reading_model_test))]

acc = reading_model.score(X_test, y_reading_test)
print(acc)

from sklearn.metrics import recall_score, precision_score, f1_score

precision = precision_score(y_reading_test, y_reading_model_test, average="weighted")
p_baseline = precision_score(y_reading_test, y_reading_baseline, average="weighted")
r_baseline = recall_score(y_reading_test, y_reading_baseline, average="weighted")
recall = recall_score(y_reading_test, y_reading_model_test, average="weighted")
f1 = f1_score(y_reading_test, y_reading_model_test, average="weighted")
f1_baseline = f1_score(y_reading_test, y_reading_baseline, average="weighted")
print(precision)
print(p_baseline)
print(recall)
print(r_baseline)
print(f1)
print(f1_baseline)

writing_model = create_random_forest_model(40)
writing_model.fit(X_train_val, y_writing_train_val)

y_writing_model_test = writing_model.predict(X_test)
y_writing_model_proba_test = writing_model.predict_proba(X_test)

y_writing_baseline = [1 for _ in range(len(y_writing_model_test))]

from sklearn.metrics import recall_score, precision_score, f1_score

precision = precision_score(y_writing_test, y_writing_model_test, average="weighted")
p_baseline = precision_score(y_writing_test, y_writing_baseline, average="weighted")
r_baseline = recall_score(y_writing_test, y_writing_baseline, average="weighted")
recall = recall_score(y_writing_test, y_writing_model_test, average="weighted")
f1 = f1_score(y_writing_test, y_writing_model_test, average="weighted")
f1_baseline = f1_score(y_writing_test, y_writing_baseline, average="weighted")
print(precision)
print(p_baseline)
print(recall)
print(r_baseline)
print(f1)
print(f1_baseline)

feature_names = ['Gender', 'Ethnicity', 'Parent Level of Education', 
                 'Lunch', 'Test Preparation']
math_fi = math_model.feature_importances_ * 100
reading_fi = reading_model.feature_importances_ * 100
writing_fi = writing_model.feature_importances_ * 100

fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(len(feature_names))
bar_width = 0.2
opacity = 0.9
error_config = {'ecolor': '0.3'}
ax.bar(index, math_fi, bar_width,
       alpha=opacity, color='r',
       label='Math Grade Model',
       hatch='x')
ax.bar(index + bar_width, reading_fi, bar_width,
       alpha=opacity, color='g',
       label='Reading Grade Model',
       hatch='o')
ax.bar(index + bar_width * 2, writing_fi, bar_width,
       alpha=opacity, color='b',
       label='Writing Grade Model',
       hatch='+')   
ax.set_xlabel('')
ax.set_ylabel('Feature Importance (%)')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(feature_names)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)

parent_codes = np.arange(0, 6)
parent_codes_decoded = [' '.join([x.capitalize() for x in p.split()]) for p in ple]
parent_title = str(dict(zip(parent_codes, parent_codes_decoded)))

race_codes = np.arange(0, 5)
race_codes_decoded = [' '.join([x.capitalize() for x in r.split()]) for r in race_le.inverse_transform(race_codes)]
race_title = str(dict(zip(race_codes, race_codes_decoded)))

parent_title = '{0: \'High School\', 1: \'Some High School\', 2: \'Some College\', 3: "Associate\'s Degree", 4: "Bachelor\'s Degree", 5: "Master\'s Degree"}'

print(f"Entering PDPIsolate")
pdp_race = pdp.PDPIsolate(model=math_model,
                          df=df,
                          model_features=['gender_le', 'race_le', 'parent_le', 'lunch_le', 'test_prep_le'],
                          feature='race_le',
                          feature_name='Race/Ethnicity')
ethnicity_xtick_labels = ['Group A', 'Group B', 'Group C',
                          'Group D', 'Group E']
plot_params = {
    # plot title and subtitle
    'title': 'PDP for Ethnicity - Math Grade',
    'subtitle': 'Race/Ethnicity (Legend): \n%s' % (race_title),
    'title_fontsize': 15,
    'subtitle_fontsize': 12,
    # color for contour line
    'contour_color':  'white',
    'font_family': 'Arial',
    # matplotlib color map for interact plot
    'cmap': 'viridis',
    # fill alpha for interact plot
    'inter_fill_alpha': 0.8,
    # fontsize for interact plot text
    'inter_fontsize': 9,
}
fig, axes = pdp_race.plot(center=False, ncols=2, plot_lines=False, frac_to_plot=100,
                         plot_params=plot_params, engine='matplotlib')
print(axes)
axes['line_axes'][0].set_xlabel('Ethnicity')
axes['line_axes'][1].set_xlabel('Ethnicity')
axes['line_axes'][2].set_xlabel('Ethnicity')
axes['line_axes'][3].set_xlabel('Ethnicity')
axes['line_axes'][0].set_title('Grade A')
axes['line_axes'][1].set_title('Grade B')
axes['line_axes'][2].set_title('Grade C')
axes['line_axes'][3].set_title('Grade F')
axes['line_axes'][0].set_xticks(race_codes)
axes['line_axes'][1].set_xticks(race_codes)
axes['line_axes'][2].set_xticks(race_codes)
axes['line_axes'][3].set_xticks(race_codes)
axes['line_axes'][0].set_xticklabels(ethnicity_xtick_labels)
axes['line_axes'][1].set_xticklabels(ethnicity_xtick_labels)
axes['line_axes'][2].set_xticklabels(ethnicity_xtick_labels)
axes['line_axes'][3].set_xticklabels(ethnicity_xtick_labels)
#fig.savefig('images/chapter_03_pdp_race_math.png', dpi=300, bbox_inches='tight')
#fig.savefig('images/chapter_03_pdp_race_math.pdf', format='pdf', dpi=300, bbox_inches='tight')

pdp_race = pdp.PDPIsolate(model=math_model,
                          df=df,
                          model_features=feature_cols,
                          feature='parent_le',
                          feature_name='Parent Level Education')
ple_xticklabels = ['High School',
                   'Some High School',
                   'Some College',
                   "Associate\'s Degree",
                   "Bachelor\'s Degree",
                   "Master\'s Degree"]
plot_params = {
    # plot title and subtitle
    'title': 'PDP for Parent Level Educations - Math Grade',
    'subtitle': 'Parent Level Education (Legend): \n%s' % (parent_title),
    'title_fontsize': 15,
    'subtitle_fontsize': 12,
    # color for contour line
    'contour_color':  'white',
    'font_family': 'Arial',
    # matplotlib color map for interact plot
    'cmap': 'viridis',
    # fill alpha for interact plot
    'inter_fill_alpha': 0.8,
    # fontsize for interact plot text
    'inter_fontsize': 9,
}
fig, axes = pdp_race.plot(center=True, ncols=2, plot_lines=False, frac_to_plot=100,
                         plot_params=plot_params, engine='matplotlib')
axes['line_axes'][0].set_xlabel('Parent Level Education')
axes['line_axes'][1].set_xlabel('Parent Level Education')
axes['line_axes'][2].set_xlabel('Parent Level Education')
axes['line_axes'][3].set_xlabel('Parent Level Education')
axes['line_axes'][0].set_title('Grade A')
axes['line_axes'][1].set_title('Grade B')
axes['line_axes'][2].set_title('Grade C')
axes['line_axes'][3].set_title('Grade F')
axes['line_axes'][0].set_xticks(parent_codes)
axes['line_axes'][1].set_xticks(parent_codes)
axes['line_axes'][2].set_xticks(parent_codes)
axes['line_axes'][3].set_xticks(parent_codes)
axes['line_axes'][0].set_xticklabels(ple_xticklabels)
axes['line_axes'][1].set_xticklabels(ple_xticklabels)
axes['line_axes'][2].set_xticklabels(ple_xticklabels)
axes['line_axes'][3].set_xticklabels(ple_xticklabels)
for tick in axes['line_axes'][0].get_xticklabels():
    tick.set_rotation(45)
for tick in axes['line_axes'][1].get_xticklabels():
    tick.set_rotation(45)
for tick in axes['line_axes'][2].get_xticklabels():
    tick.set_rotation(45)
for tick in axes['line_axes'][3].get_xticklabels():
    tick.set_rotation(45)
#fig.savefig('images/chapter_03_pdp_parent_math.png', dpi=300, bbox_inches='tight')
#fig.savefig('images/chapter_03_pdp_parent_math.pdf', format='pdf', dpi=300, bbox_inches='tight')

pdp_race_parent = pdp.PDPInteract(model=math_model,
                                   df=df,
                                   model_features=feature_cols,
                                   features=['race_le', 'parent_le'],
                                   feature_names=['Race/Ethnicity', 'Parent Level of Education'])
plot_params = {
    # plot title and subtitle
    'title': 'PDP Interaction - Math Grade',
    'subtitle': 'Race/Ethnicity (Legend): \n%s\nParent Level of Education (Legend): \n%s' % (race_title, parent_title),
    'title_fontsize': 15,
    'subtitle_fontsize': 12,
    # color for contour line
    'contour_color':  'white',
    'font_family': 'Arial',
    # matplotlib color map for interact plot
    'cmap': 'viridis',
    # fill alpha for interact plot
    'inter_fill_alpha': 0.8,
    # fontsize for interact plot text
    'inter_fontsize': 9,
}
fig, axes = pdp_race_parent.plot(plot_type='grid', plot_pdp=True, plot_params=plot_params, engine='plotly')
try:
    # Display Plotly figure in an interactive window/browser
    fig.show()
except Exception as _:
    pass
try:
    # Also save an HTML copy as a fallback you can open manually
    fig.write_html('pdp_interaction_race_parent_grid.html')
except Exception as _:
    pass


if axes:
    axes['interact_axes'][0].set_title('Grade A')
    axes['interact_axes'][1].set_title('Grade B')
    axes['interact_axes'][2].set_title('Grade C')
    axes['interact_axes'][3].set_title('Grade F')
    axes['interact_axes'][0].grid(False)
    axes['interact_axes'][1].grid(False)
    axes['interact_axes'][2].grid(False)
    axes['interact_axes'][3].grid(False)
    axes['isolate_axes'][0][0].set_title('Grade A - Race/Ethnicity PDP')
    axes['isolate_axes'][0][1].set_title('Grade A - Parent Education PDP')
    axes['isolate_axes'][1][0].set_title('Grade B - Race/Ethnicity PDP')
    axes['isolate_axes'][1][1].set_title('Grade B - Parent Education PDP')
    axes['isolate_axes'][2][0].set_title('Grade C - Race/Ethnicity PDP')
    axes['isolate_axes'][2][1].set_title('Grade C - Parent Education PDP')
    axes['isolate_axes'][3][0].set_title('Grade F - Race/Ethnicity PDP')
    axes['isolate_axes'][3][1].set_title('Grade F - Parent Education PDP')
    axes['isolate_axes'][0][0].grid(False)
    axes['isolate_axes'][0][1].grid(False)
    axes['isolate_axes'][1][0].grid(False)
    axes['isolate_axes'][1][1].grid(False)
    axes['isolate_axes'][2][0].grid(False)
    axes['isolate_axes'][2][1].grid(False)
    axes['isolate_axes'][3][0].grid(False)
    axes['isolate_axes'][3][1].grid(False)
#fig.savefig('images/chapter_03_pdp_interaction_race_parent_math.png', dpi=300, bbox_inches='tight');
#fig.savefig('images/chapter_03_pdp_interaction_race_parent_math.pdf', format='pdf', dpi=300, bbox_inches='tight');

