import gitlab
import json
import time 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--task-json", type=str, default='run.json')
parser.add_argument("--pipeline-id", type=int, default=None)
args = parser.parse_args()

def check_jobs_status(jobs):
    status = 'success'
    for job in jobs:
        if job.stage in ('.pre', 'build'):
            if job.status in ('failed', 'canceled'):
                return 'failed'
            elif job.status != 'success':
                status = 'running'
    return status


# https://vg-code.gml-team.ru/-/profile/personal_access_tokens
private_token = "private_token"

gl = gitlab.Gitlab(url='https://vg-code.gml-team.ru/', private_token=private_token, retry_transient_errors=True)

project_name_with_namespace = "framework/metrics"
project = gl.projects.get(project_name_with_namespace)


with open(args.task_json) as json_file:
    task = json.load(json_file)

if args.pipeline_id is None:
    variables = [{'key': attack.replace('-', '_'), 'value': 'yes'} for attack in task.keys()]
    
    pipeline = project.pipelines.create({'ref': 'master', 'variables': variables})
    print(f'Pipeline {pipeline.id} created')
    print(pipeline.web_url)
    
    print('Waiting 10 min for metrics to be built')
    time.sleep(10 * 60)
else:
    pipeline = project.pipelines.get(args.pipeline_id)
    print(f'Pipeline {args.pipeline_id} found')
    print(pipeline.web_url)
    
    
retrying_n = 0
while True:
    jobs = pipeline.jobs.list(iterator=True)
    jobs_status = check_jobs_status(jobs)
    if jobs_status == 'success':
        print('Metrics builded successful')
        break
    elif jobs_status == 'failed':
        if retrying_n == 5:
            print('Failed to build metrics. Check the logs! Exit')
            exit(0)
        print(f'Building failed. Retrying #{retrying_n}')
        pipeline.retry()
        retrying_n += 1
    print('Waiting 2 min for metrics to be built')
    time.sleep(2 * 60)
    

bridges = pipeline.bridges.list(iterator=True)
need_wait = False
for bridge in bridges:
    if bridge.name in task.keys() and bridge.downstream_pipeline is None:
        need_wait = True
if need_wait:
    print('Downstream pipelines are not ready. Waiting 1 min')
    time.sleep(1 * 60)
    
print('Running jobs')
while True:
    retry = False
    bridges = pipeline.bridges.list(iterator=True)
    for bridge in bridges:
        print(bridge.name)
        if bridge.name in task.keys():
            if bridge.downstream_pipeline is None:
                print(f'Downstream pipeline for {bridge.name} was not created. Run it manually and rerun script')
            downstream_pipeline_id = bridge.downstream_pipeline['id']
            downstream_pipeline = project.pipelines.get(downstream_pipeline_id)
            jobs = downstream_pipeline.jobs.list(iterator=True)
            for job in jobs:
                metric = job.name.split(':')[0].split('_')[1]
                if job.stage == 'build' and metric in task[bridge.name]:
                    if job.status == 'manual':
                        real_job = project.jobs.get(job.id)
                        real_job.play()
                    elif job.status == 'created':
                        retry = True
    if not retry:
        break
    print('Some jobs are not ready to run. Waiting 1 min and retrying')
    time.sleep(1 * 60)
        

