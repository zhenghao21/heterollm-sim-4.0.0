"""No GPU execution; regression checks for evidence attribution and overlap."""
import unittest
from analyze import union_ns, attributed_kernels

class AttributionTests(unittest.TestCase):
    def test_overlapping_kernel_intervals_count_once(self):
        self.assertEqual(union_ns([(0,10),(5,15),(20,25),(25,30)]),25)
        self.assertEqual(union_ns([]),0)

    def test_backwards_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            union_ns([(11,10)])

    def test_api_correlation_pid_and_issuing_thread_all_match(self):
        pid=7<<24; tid=pid+11
        marker={'start':100,'end':200,'globalTid':tid}
        apis=[{'start':110,'end':120,'globalTid':tid,'correlationId':4},
              {'start':120,'end':130,'globalTid':tid+1,'correlationId':5},
              {'start':90,'end':99,'globalTid':tid,'correlationId':6}]
        kernels=[{'globalPid':pid,'correlationId':4},
                 {'globalPid':8<<24,'correlationId':4},
                 {'globalPid':pid,'correlationId':5},
                 {'globalPid':pid,'correlationId':6}]
        matched_api,matched_kernel=attributed_kernels(marker,apis,kernels)
        self.assertEqual(matched_api,[apis[0]])
        self.assertEqual(matched_kernel,[kernels[0]])

if __name__=='__main__':
    unittest.main()
