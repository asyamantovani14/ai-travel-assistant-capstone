import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from run_daily_pipeline import PROJECT_ROOT, STEPS, run_step, select_steps


def test_project_root_contains_src_directory():
    assert (PROJECT_ROOT / "src").is_dir()


def test_select_steps_supports_resume_and_single_step():
    assert select_steps(only="evaluation") == ["evaluation"]
    assert select_steps(from_step="clean_articles") == list(STEPS[2:])


def test_select_steps_rejects_unknown_step():
    with pytest.raises(ValueError, match="Unknown pipeline step"):
        select_steps(only="publish_everything")


@patch("run_daily_pipeline.subprocess.run")
def test_run_step_uses_project_root_as_working_directory(mock_run: Mock):
    mock_run.return_value = subprocess.CompletedProcess([], 0, "done", "")

    run_step("evaluation")

    command = mock_run.call_args.args[0]
    assert Path(command[1]) == PROJECT_ROOT / "src" / "evaluation.py"
    assert mock_run.call_args.kwargs["cwd"] == PROJECT_ROOT


@patch("run_daily_pipeline.subprocess.run")
def test_dry_run_does_not_start_a_process(mock_run: Mock):
    run_step("evaluation", dry_run=True)
    mock_run.assert_not_called()


@patch("run_daily_pipeline.subprocess.run")
def test_run_step_propagates_failure(mock_run: Mock):
    mock_run.return_value = subprocess.CompletedProcess([], 7, "", "failed")

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_step("evaluation")

    assert error.value.returncode == 7
