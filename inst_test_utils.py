#%%
import json
import sqlite3

import time

import json
import subprocess



class PersistentDict:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS dictionary (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.commit()

    def __getitem__(self, key):
        cursor = self.conn.execute("SELECT value FROM dictionary WHERE key = ?", (key,))
        result = cursor.fetchone()
        if result is not None:
            return json.loads(result[0])
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        value = json.dumps(value)
        self.conn.execute("REPLACE INTO dictionary (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def __delitem__(self, key):
        self.conn.execute("DELETE FROM dictionary WHERE key = ?", (key,))
        self.conn.commit()

    def __iter__(self):
        cursor = self.conn.execute("SELECT key FROM dictionary")
        for row in cursor:
            yield row[0]

    def __len__(self):
        cursor = self.conn.execute("SELECT COUNT(*) FROM dictionary")
        return cursor.fetchone()[0]

    def items(self):
        cursor = self.conn.execute("SELECT key, value FROM dictionary")
        for row in cursor:
            key = row[0]
            value = json.loads(row[1])
            yield key, value

    def values(self):
        cursor = self.conn.execute("SELECT value FROM dictionary")
        for row in cursor:
            yield json.loads(row[0])


class SlurmSubmissionsDb():
    def __init__(self, db_name):
        self._pd = PersistentDict(db_name)
        

    def add_submission(self, name, qubit_count, orig_circ_name, path_to_block, inst_name, dist_tol, slurm_job_id, slurm_log_file_path, inst_params:dict):
        d = {'qubit_count': qubit_count,
            'orig_circ_name': orig_circ_name,
            'path_to_block': path_to_block,
            'inst_name': inst_name,
            'dist_tol': dist_tol,
            'slurm_job_id':slurm_job_id,
            'slurm_log_file_path':slurm_log_file_path,
            'submit time':time.ctime(),
            'is_done':False}
        d.update(inst_params)
        self._pd[name] = d

    @staticmethod
    def _retrive_job_running_status(jobid):

        command = f"sacct -j {jobid} | grep gpu_ss1 | awk '{{print $6}}'"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print(f"Error occurred: {error}")
        return output.decode('utf-8').strip()

    @staticmethod
    def _parse_slurm_log(path_to_log):
        with open(path_to_log, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if len(line) > 3 and line[:3] =='***':
                break
        else:
            print(f"Error - couldn't find result in {path_to_log}")
            raise(RuntimeError(f"Error - couldn't find result in {path_to_log}"))
        
        return json.loads(line[3:])
        

    def get_submission(self, name):
        record = self._pd[name]
        
        if record['is_done']:
            return record
        
        jobid = record['slurm_job_id']
        job_status = SlurmSubmissionsDb._retrive_job_running_status(jobid)
        record['job_status'] = job_status

        if job_status == 'COMPLETED':
            inst_time, final_diff_to_target = SlurmSubmissionsDb._parse_slurm_log(record['slurm_log_file_path'])
            record['inst_time'] = inst_time
            record['final_diff_to_target'] = final_diff_to_target
            record['inst_succsed'] = final_diff_to_target < record['dist_tol']
            record['is_done'] = True
        elif job_status in ['TIMEOUT', 'OUT_OF_ME+']:
            record['inst_succsed'] = False
            record['is_done'] = True            
        
        return record

    def get_all_submissions(self):
        for key in self._pd:
            yield self.get_submission(key)

    def exists(self, name):
        return name in self._pd