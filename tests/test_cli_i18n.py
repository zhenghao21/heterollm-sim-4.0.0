"""CLI 中文提示契约测试。"""

from contextlib import redirect_stderr
from io import StringIO
import unittest

from heterollm_sim.cli import _exception_message_zh, build_parser


class CliChineseMessageTests(unittest.TestCase):
    def test_root_help_uses_chinese_headings_and_copy(self):
        help_text = build_parser().format_help()
        self.assertIn("用法：", help_text)
        self.assertIn("位置参数：", help_text)
        self.assertIn("选项：", help_text)
        self.assertIn("显示帮助并退出", help_text)
        self.assertNotIn("usage:", help_text)
        self.assertNotIn("show this help message", help_text)

    def test_parse_error_is_chinese(self):
        stream = StringIO()
        with redirect_stderr(stream), self.assertRaises(SystemExit):
            build_parser().parse_args([])
        message = stream.getvalue()
        self.assertIn("参数错误", message)
        self.assertIn("缺少必填参数", message)
        self.assertNotIn("the following arguments are required", message)

    def test_unknown_english_exception_uses_chinese_fallback(self):
        message = _exception_message_zh(ValueError("capacity_bytes must be positive"))
        self.assertIn("操作失败", message)
        self.assertNotIn("must be positive", message)

    def test_existing_chinese_exception_keeps_specific_detail(self):
        self.assertEqual(
            _exception_message_zh(ValueError("容量必须为非负数。")),
            "容量必须为非负数。",
        )


if __name__ == "__main__":
    unittest.main()
