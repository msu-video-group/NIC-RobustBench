import gitlab
import subprocess
import pandas as pd

# https://vg-code.gml-team.ru/-/profile/personal_access_tokens
private_token = "private_token"
url = 'https://vg-code.gml-team.ru/'
gl = gitlab.Gitlab(url=url, private_token=private_token)


with open('../scripts/metrics.txt') as f:
    metrics = f.read().splitlines()
with open('../scripts/methods.txt') as f:
    methods = f.read().splitlines()
    with open('../scripts/trainable_methods.txt') as f:
        trainable_methods = f.read().splitlines()
columns = []
for i, method in enumerate(methods):
    columns.append((method, 'build'))
    if method in trainable_methods:
        columns.append((method, 'train'))
    columns.append((method, 'test'))

df = pd.DataFrame(index=metrics, columns=pd.MultiIndex.from_tuples(columns, names=['method', 'stage']), dtype=pd.StringDtype())


project_name_with_namespace = "framework/metrics"
project = gl.projects.get(project_name_with_namespace)

jobs = project.jobs.list(iterator=True, order_by='created_at', sort='desc')
for job in jobs:
    if job.created_at < '2023-06-18':
        break
    if job.status == 'success':
        job_id = job.id
        if ':' not in job.name:
            continue
        method_metric, stage = job.name.split(':')
        if '_' not in method_metric:
            continue
        print(method_metric)
        method, metric = method_metric.split('_')
        if metric in df.index and (method, stage) in df.columns:
            if pd.isna(df.loc[metric, (method, stage)]):
                df.loc[metric, (method, stage)] =f'=HYPERLINK("{url}{project_name_with_namespace}/-/jobs/{job_id}/", "{job_id}")'
df.to_excel("results.xlsx")  