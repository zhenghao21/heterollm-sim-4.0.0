"""No GPU execution: destructive timeout and freeze rejection regressions."""
import subprocess
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json
import pilot
from pilot import wait_without_killing
from inspect_schema import describe_schema
import sqlite3

class LifecycleTests(unittest.TestCase):
    def test_timeout_preserves_owned_process_and_exposes_status(self):
        process = Mock()
        process.wait.side_effect = subprocess.TimeoutExpired(['synthetic'], 180)
        receipt = {'pid': 42}
        result = wait_without_killing(process, receipt)
        self.assertEqual(result['pid'], 42)
        self.assertEqual(result['status'], 'still_running')
        self.assertIsNone(result['returncode'])
        process.kill.assert_not_called()
        process.terminate.assert_not_called()

    def test_success_and_failure_keep_exit_codes(self):
        for code, status in ((0,'completed'), (7,'failed')):
            process = Mock()
            process.wait.return_value = code
            result = wait_without_killing(process, {'pid': 42})
            self.assertEqual(result['status'], status)
            self.assertEqual(result['returncode'], code)
            process.kill.assert_not_called()

    def test_empty_freeze_file_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / 'freeze_manifest.json').write_text(json.dumps({'files': []}))
            with patch.object(pilot, 'HERE', target):
                with self.assertRaisesRegex(ValueError, 'empty freeze file map'):
                    pilot.verify_freeze()

    def test_schema_inventory_records_unknown_names_without_guessing(self):
        with sqlite3.connect(':memory:') as connection:
            connection.execute('CREATE TABLE "new_table" ("strange" INTEGER)')
            observed = describe_schema(connection)
        self.assertEqual(observed['tables']['new_table'][0]['name'], 'strange')
        self.assertFalse(observed['column_mappings_guessed'])

if __name__ == '__main__':
    unittest.main()
