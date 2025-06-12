import gitlab
from collections import defaultdict, Counter
import click

# https://vg-code.gml-team.ru/-/profile/personal_access_tokens
private_token = "PRIVATE_TOKEN"
pipeline_id = 10636


gl = gitlab.Gitlab(url='https://vg-code.gml-team.ru/', private_token=private_token)

project_name_with_namespace = "framework/metrics"
project = gl.projects.get(project_name_with_namespace)



pipeline = project.pipelines.get(pipeline_id)


jobs = pipeline.jobs.list(get_all=True)


bridges = pipeline.bridges.list(get_all=True)

def print_dict(dct):
    ordered_stages = ['build', 'train', 'test', 'collect']
    sorted_dct = {k: dct[k] for k in ordered_stages if k in dct}
    for stage, statuses in sorted_dct.items():
        print(f'\t{stage}:')
        for status, n in statuses.items():    
            print('\t\t', status, n)
        
if click.confirm('Do you want to retry failed jobs?', default=False):
    retry = True
else:
    retry = False

print()

statuses = defaultdict(lambda: defaultdict(int))
for bridge in bridges:
    print(bridge.name)
    if bridge.downstream_pipeline is None:
        print('\tNot running')
        print()
        continue
    downstream_pipeline_id = bridge.downstream_pipeline['id']
    downstream_pipeline = project.pipelines.get(downstream_pipeline_id)
    if retry:
        downstream_pipeline.retry()
    downstream_pipeline_jobs = downstream_pipeline.jobs.list(get_all=True)
    downstream_statuses = defaultdict(lambda: defaultdict(int))
    for downstream_job in downstream_pipeline_jobs:
        downstream_statuses[downstream_job.stage][downstream_job.status] += 1
    print_dict(downstream_statuses)
    print()
    for stage, status_dict in downstream_statuses.items():
        for status, n in status_dict.items():
            statuses[stage][status] += n


print('Summary')
print_dict(statuses)