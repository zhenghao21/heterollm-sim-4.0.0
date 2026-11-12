"""Audit-copy sidecar: component history and genuinely committed observations.

Channel images are immutable between calls. Savepoints copy their small index,
not command events; images contain every rule's bounded history. Component files
are cumulative committed views, so restoring a public snapshot also restores
the output. Workers never publish files while a physical transaction is open.
"""
from __future__ import annotations
import copy
import itertools
import os
from pathlib import Path
import numpy as np

_identities=itertools.count()
_types=('ACT','PRE','PRE_ALL','REFab','REFpb','RD','WR')


class AuditState:
    def __init__(self):
        self.identity=f'{os.getpid()}_{next(_identities)}'
        self.summary_prefix=os.environ.get('HETEROLLM_HBM_TRACE_SUMMARY_FILE')
        self.trace_prefix=os.environ.get('HETEROLLM_HBM_TRACE_FILE')
        self.stride=0
        self.history={}
        self.counts=dict.fromkeys(('total',*_types,'unaudited_batches','unaudited_requests'),0)
        self.checks=[0]*35
        self.violations=[0]*35
        self.first=[]
        self.trace=[]
        self.stack=[]

    def configure(self,g):
        from ._hbm_fr_kernel import audit_image_size
        stride=audit_image_size(int(g[1]),int(g[5]),int(g[14]))
        if self.stride and self.stride!=stride:
            raise ValueError('Audit geometry changed within a physical component')
        self.stride=stride

    def checkpoint(self):
        return (self.history.copy(),self.counts.copy(),self.checks.copy(),
                self.violations.copy(),self.first.copy(),self.trace.copy())

    def restore(self,save):
        self.history,self.counts,self.checks,self.violations,self.first,self.trace=save

    def begin(self,token):
        self.stack.append((id(token),self.checkpoint()))

    def commit(self,token):
        identity,_=self.stack.pop()
        assert identity==id(token)
        self.publish()

    def rollback(self,token):
        identity,save=self.stack.pop()
        assert identity==id(token)
        self.restore(save)
        self.publish()

    def images(self,left,right):
        result=np.zeros((right-left,self.stride),dtype=np.uint8)
        for channel in range(left,right):
            saved=self.history.get(channel)
            if saved is not None:
                result[channel-left]=np.frombuffer(saved,dtype=np.uint8)
        return result

    def consume(self,left,images,trace,summary):
        for index,row in enumerate(images):
            self.history[left+index]=row.tobytes()
        if trace and any(not line.startswith(b'#') for line in trace.splitlines()):
            self.trace.append(trace)
        for line in summary.decode('utf-8').splitlines():
            parts=line.split('\t')
            fields=dict(part.split('=',1) for part in parts[1:] if '=' in part)
            if parts[0]=='SUMMARY':
                for key,value in fields.items():
                    self.counts[key]+=int(value)
            elif parts[0]=='AUDIT_DELTA':
                counts=list(map(int,fields['checks'].split(',')))
                violations=list(map(int,fields['violations'].split(',')))
                assert len(counts)==len(violations)==35
                self.checks=[a+b for a,b in zip(self.checks,counts)]
                self.violations=[a+b for a,b in zip(self.violations,violations)]
                if fields.get('first'):
                    self.first.append(fields['first'])

    def output_path(self,prefix):
        return Path(f'{prefix}.component_{self.identity}')

    def publish(self):
        if self.stack:
            return
        if self.summary_prefix:
            text='#COMMITTED_COMPONENT\tidentity='+self.identity+'\thistory=channel_images_v2\n'
            text+='SUMMARY\t'+'\t'.join(f'{name}={value}' for name,value in self.counts.items())+'\n'
            text+=f'AUDIT_DELTA\tcommands={self.counts["total"]}\tchecks='+','.join(map(str,self.checks))
            text+='\tviolations='+','.join(map(str,self.violations))+'\tfirst='+';'.join(self.first)+'\n'
            self.output_path(self.summary_prefix).write_text(text,encoding='utf-8')
        if self.trace_prefix:
            self.output_path(self.trace_prefix).write_bytes(b''.join(self.trace))

    def remove_output(self):
        for prefix in (self.summary_prefix,self.trace_prefix):
            if prefix:
                self.output_path(prefix).unlink(missing_ok=True)


def audit_for(core):
    if core.config.control_mode!='jedec_hbm' or not (
            os.environ.get('HETEROLLM_HBM_TRACE_SUMMARY_FILE') or os.environ.get('HETEROLLM_HBM_TRACE_FILE')):
        return None
    audit=getattr(core,'_hbm_audit',None)
    if audit is None:
        audit=core._hbm_audit=AuditState()
    return audit
