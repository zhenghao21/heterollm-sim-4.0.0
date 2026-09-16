"""Deterministic hostmock tests; no Win32 timer or NVML is created."""
import threading
import unittest
from sampler import DeadlineGrid, sample_loop, SamplerSession
from assess import summarize, formal_clock_gate

class FakeTimer:
    def __init__(self, lateness=0):
        self.t=1000;self.frequency=1_000_000_000;self.stopped=False;self.closed=False;self.lateness=lateness;self.targets=[]
        self.metadata={'backend':'hostmock','global_timer_resolution_changed':False}
    def now(self):return self.t
    def wait_until(self,deadline):
        self.targets.append(deadline);begin=self.t
        if self.stopped:return {'status':'stopped','wait_begin_qpc':begin,'wait_end_qpc':self.t,'arms':[]}
        self.t=max(self.t,deadline)+self.lateness
        return {'status':'deadline_reached','wait_begin_qpc':begin,'wait_end_qpc':self.t,'arms':[]}
    def signal_stop(self):self.stopped=True
    def close(self):self.closed=True

class FakeReader:
    def __init__(self,timer,cost=100_000,value=2392,status=0):
        self.timer=timer;self.cost=cost;self.value=value;self.status=status;self.closed=False
        self.identity={'uuid':'hostmock','clock_writes_performed':False}
    def read_sm(self):
        if self.closed:raise RuntimeError('closed reader')
        self.timer.t+=self.cost
        return {'status':self.status,'value':self.value if self.status==0 else None,'unit':'MHz'}
    def close(self):self.closed=True

class DeadlineTests(unittest.TestCase):
    def test_absolute_grid_no_readcost_drift(self):
        timer=FakeTimer();reader=FakeReader(timer,cost=2_000_000)
        raw=sample_loop(timer,reader,duration_ns=2_000_000_000)
        self.assertEqual(len(raw['samples']),400);self.assertEqual(raw['missed_deadlines'],[])
        self.assertTrue(all(b-a==5_000_000 for a,b in zip(timer.targets,timer.targets[1:])))
        summary=summarize(raw);self.assertEqual(summary['sample_start_intervals_ms']['median'],5.)
        self.assertEqual(summary['SM_read_cost_ms']['median'],2.)
    def test_fractional_QPC_tick_grid_uses_ceil_without_drift(self):
        grid=DeadlineGrid(1,1024,5_000_000)
        self.assertEqual(grid.deadline(200),1025)
        for i in range(1,50):self.assertEqual(grid.next_future_index(grid.deadline(i),i),i+1)
    def test_overrun_skip_instead_of_catchup_burst(self):
        timer=FakeTimer();reader=FakeReader(timer,cost=7_000_000)
        raw=sample_loop(timer,reader,duration_ns=40_000_000)
        self.assertEqual([s['index'] for s in raw['samples']],[0,2,4,6])
        self.assertEqual(summarize(raw)['deadline_slots_missed'],4)
    def test_late_wake_beyond_window_not_saved_as_a_sample(self):
        timer=FakeTimer(lateness=30_000_000);reader=FakeReader(timer)
        raw=sample_loop(timer,reader,duration_ns=20_000_000)
        self.assertEqual(raw['samples'],[]);self.assertEqual(raw['missed_deadlines'][0]['reason'],'wake_after_measurement_window')
    def test_stop_prevents_due_sample(self):
        timer=FakeTimer();timer.signal_stop()
        raw=sample_loop(timer,FakeReader(timer),duration_ns=10_000_000)
        self.assertEqual(raw['samples'],[]);self.assertEqual(raw['waits'][0]['status'],'stopped')
    def test_read_errors_retained(self):
        timer=FakeTimer();reader=FakeReader(timer,status=99)
        raw=sample_loop(timer,reader,duration_ns=10_000_000)
        self.assertEqual(len(raw['samples']),2);self.assertIn('SM_readback_failure',summarize(raw)['errors'])
    def test_explicit_read_begin_end(self):
        timer=FakeTimer();raw=sample_loop(timer,FakeReader(timer,cost=300_000),duration_ns=10_000_000)
        for s in raw['samples']:self.assertEqual(s['sm_read_end_qpc']-s['sm_read_begin_qpc'],300_000)
    def test_no_inference_means_no_formal_validation_claim(self):
        timer=FakeTimer();raw=sample_loop(timer,FakeReader(timer),duration_ns=10_000_000)
        self.assertEqual(summarize(raw)['formal_clock_gate']['status'],'not_evaluated')
        self.assertFalse(raw['cost_calibration_eligible'])

class LifecycleTests(unittest.TestCase):
    def test_shutdown_only_after_worker_exits(self):
        timer=FakeTimer();reader=FakeReader(timer)
        session=SamplerSession(timer=timer,reader_factory=lambda:reader,duration_ns=10_000_000).start()
        session.join(1);self.assertTrue(reader.closed);self.assertFalse(timer.closed)
        session.close();self.assertTrue(timer.closed);self.assertFalse(session.thread.is_alive())
    def test_join_timeout_does_not_close_live_resources(self):
        timer=FakeTimer();entered=threading.Event();release=threading.Event()
        class BlockingReader(FakeReader):
            def read_sm(self):entered.set();release.wait(2);return super().read_sm()
        reader=BlockingReader(timer)
        session=SamplerSession(timer=timer,reader_factory=lambda:reader,duration_ns=10_000_000).start()
        self.assertTrue(entered.wait(1))
        try:
            with self.assertRaises(TimeoutError):session.join(.01)
            self.assertFalse(reader.closed);self.assertFalse(timer.closed)
            with self.assertRaises(RuntimeError):session.close()
        finally:
            session.stop();release.set();session.join(1);session.close()
        self.assertTrue(reader.closed);self.assertTrue(timer.closed)
    def test_factory_failure_gets_finished_error(self):
        timer=FakeTimer()
        def fail():raise RuntimeError('mock initialization failed')
        session=SamplerSession(timer=timer,reader_factory=fail).start();session.join(1);session.close()
        self.assertTrue(session.errors);self.assertTrue(session.done.is_set());self.assertTrue(timer.closed)

class FormalGateTests(unittest.TestCase):
    def make(self,value=2392):
        timer=FakeTimer();return sample_loop(timer,FakeReader(timer,value=value),duration_ns=100_000_000)
    def test_2392_satisfies_frozen_domain_with_real_read_windows(self):
        raw=self.make();origin=raw['origin_qpc'];gate=formal_clock_gate(raw,[{'index':0,'qpc_start':origin+25_200_000,'qpc_end':origin+25_300_000}])
        self.assertTrue(gate['passed'])
    def test_clock_value_outside_domain(self):
        raw=self.make(value=2000);o=raw['origin_qpc']
        self.assertIn('SM_clock_outside_target_domain',formal_clock_gate(raw,[{'index':0,'qpc_start':o+25_200_000,'qpc_end':o+25_300_000}])['issues'])
    def test_bracket_25ms_does_not_get_relaxed(self):
        raw=self.make();o=raw['origin_qpc'];raw['samples']=[raw['samples'][0],raw['samples'][15]]
        self.assertIn('formal_clock_bracket_too_wide',formal_clock_gate(raw,[{'index':0,'qpc_start':o+7_000_000,'qpc_end':o+8_000_000}])['issues'])
        with self.assertRaises(ValueError):formal_clock_gate(raw,[],maximum_gap_ms=30)
    def test_no_silent_single_timestamp_substitution(self):
        raw=self.make();del raw['samples'][0]['sm_read_begin_qpc']
        o=raw['origin_qpc'];self.assertFalse(formal_clock_gate(raw,[{'index':0,'qpc_start':o+1_000_000,'qpc_end':o+2_000_000}])['passed'])
    def test_missing_formal_intervals_not_passed(self):self.assertFalse(formal_clock_gate(self.make(),[])['passed'])

if __name__=='__main__':unittest.main(verbosity=2)
