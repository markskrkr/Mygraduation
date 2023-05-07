# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:49:37 2023

@author: DELL
"""

import heapq

def schedule_jobs(jobs, manual_jobs):
    sorted_jobs = sorted(jobs, key=lambda x: x["release_date"])
    job_queue = []
    current_day = 0
    max_difference = -1
    work_intervals = {job["name"]: [] for job in jobs}

    for manual_job in manual_jobs:
        job_index = [j["name"] for j in jobs].index(manual_job)
        job = jobs[job_index]
        current_day = max(current_day, job["release_date"]) + job["duration"]
        work_intervals[job["name"]].append((max(current_day - job["duration"], job["release_date"]), current_day))
        difference = current_day - job["due_date"]
        max_difference = max(max_difference, difference)

    sorted_jobs = [job for job in sorted_jobs if job["name"] not in manual_jobs]

    for job in sorted_jobs:
        while current_day < job["release_date"]:
            if job_queue:
                next_job = heapq.heappop(job_queue)[1]
                work_days = min(job["release_date"] - current_day, next_job["remaining"])
                start_day = current_day
                current_day += work_days
                next_job["remaining"] -= work_days
                work_intervals[next_job["name"]].append((start_day, current_day))

                if next_job["remaining"] > 0:
                    heapq.heappush(job_queue, (next_job["due_date"], next_job))
                else:
                    difference = current_day - next_job["due_date"]
                    max_difference = max(max_difference, difference)
            else:
                current_day = job["release_date"]

        heapq.heappush(job_queue, (job["due_date"], {"remaining": job["duration"], "due_date": job["due_date"], "name": job["name"]}))

    while job_queue:
        next_job = heapq.heappop(job_queue)[1]
        start_day = current_day
        current_day += next_job["remaining"]
        work_intervals[next_job["name"]].append((start_day, current_day))
        difference = current_day - next_job["due_date"]
        max_difference = max(max_difference, difference)

    return max_difference, work_intervals

jobs = [
    {"name": "a", "release_date": 0, "duration": 6, "due_date": 27},
    {"name": "b", "release_date": 2, "duration": 18, "due_date": 22},
    {"name": "c", "release_date": 14, "duration": 10, "due_date": 23},
    {"name": "d", "release_date": 27, "duration": 17, "due_date": 61},
    {"name": "e", "release_date": 40, "duration": 16, "due_date": 59},
]

manual_jobs = ["b","c","a"]
max_difference, work_intervals = schedule_jobs(jobs, manual_jobs)
print("最大日期差异：", max_difference)
print("工作间隔：")
for job_name, intervals in work_intervals.items():
    if intervals:
        total_duration = intervals[-1][1] - intervals[0][0]
        original_duration = [job["duration"] for job in jobs if job["name"] == job_name][0]
        if total_duration > original_duration:
            print(f"{job_name}: 不连续")
            print(f"{job_name}: {intervals}")
        else:
            print(f"{job_name}: {intervals}")