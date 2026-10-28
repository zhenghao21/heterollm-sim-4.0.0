"""PP conditional forecast cache identity and runner output contract."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_fast_pp_uses_forecast_contract_and_pp1_keeps_stationary_contract():
    runner = importlib.import_module("run_exploring_hbf_native")
    case = runner.case_matrix()[0]
    pp = runner.make_layout(1, 4, 2, 1)
    mb = runner.make_layout(1, 8, 1, 1, microbatches=2)
    pp_params = runner.parameters(case, pp, 4, "fast")
    mb_params = runner.parameters(case, mb, 4, "fast")
    pp_exact = runner.parameters(case, pp, 4, "detailed")
    pp1 = runner.parameters(case, runner.make_layout(1, 8, 1, 1), 4, "fast")
    assert pp_params["pipeline_execution_contract"] == runner.PIPELINE_FORECAST_CONTRACT
    assert mb_params["pipeline_execution_contract"] == runner.PIPELINE_FORECAST_CONTRACT
    assert pp_exact["pipeline_execution_contract"] == runner.PIPELINE_EXECUTION_CONTRACT
    assert "pipeline_execution_contract" not in pp1

